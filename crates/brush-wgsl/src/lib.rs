//! Procedural macro for generating WGSL kernel wrappers.
//!
//! # Usage
//!
//! Imports from the same directory are auto-discovered:
//!
//! ```ignore
//! #[wgsl_kernel(source = "src/shaders/rasterize.wgsl")]
//! pub struct Rasterize {
//!     pub bwd_info: bool,
//!     pub webgpu: bool,
//! }
//! ```
//!
//! For cross-crate imports, use explicit `includes`:
//!
//! ```ignore
//! #[wgsl_kernel(
//!     source = "src/shaders/shader.wgsl",
//!     includes = ["../other-crate/src/shaders/helpers.wgsl"],
//! )]
//! pub struct MyKernel;
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use naga::valid::Capabilities;
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use regex::Regex;
use syn::{
    Expr, ExprLit, Fields, ItemStruct, Lit, Meta, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    spanned::Spanned,
};
use wgpu::naga::{self, Handle, Type, common::wgsl::TypeContext, proc::GlobalCtx};

struct WgslKernelArgs {
    source: String,
    includes: Vec<String>,
}

impl Parse for WgslKernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut source = None;
        let mut includes = Vec::new();

        for meta in Punctuated::<Meta, Token![,]>::parse_terminated(input)? {
            match &meta {
                Meta::NameValue(nv) if nv.path.is_ident("source") => {
                    source = Some(expect_string_lit(&nv.value)?);
                }
                Meta::NameValue(nv) if nv.path.is_ident("includes") => {
                    if let Expr::Array(arr) = &nv.value {
                        for elem in &arr.elems {
                            includes.push(expect_string_lit(elem)?);
                        }
                    } else {
                        return Err(syn::Error::new(
                            nv.value.span(),
                            "expected array of strings",
                        ));
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        meta.span(),
                        "expected `source` or `includes`",
                    ));
                }
            }
        }

        Ok(Self {
            source: source.ok_or_else(|| syn::Error::new(input.span(), "missing `source`"))?,
            includes,
        })
    }
}

fn expect_string_lit(expr: &Expr) -> syn::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) => Ok(s.value()),
        _ => Err(syn::Error::new(expr.span(), "expected string literal")),
    }
}

fn extract_import_names(source: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"#import\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?;?")
            .unwrap()
    });
    re.captures_iter(source).map(|c| c[1].to_string()).collect()
}

fn file_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_owned()
}

const DECORATION_PRE: &str = "X_naga_oil_mod_X";
const DECORATION_POST: &str = "X";

fn undecorate_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"(\x1B\[\d+\w)?([\w\d_]+){}([A-Z0-9]*){}",
            regex_syntax::escape(DECORATION_PRE),
            regex_syntax::escape(DECORATION_POST)
        ))
        .unwrap()
    })
}

fn decode_base32(s: &str) -> String {
    String::from_utf8(data_encoding::BASE32_NOPAD.decode(s.as_bytes()).unwrap()).unwrap()
}

fn demangle_name(mangled: &str) -> String {
    let demangled = undecorate_regex().replace_all(mangled, |caps: &regex::Captures| {
        format!(
            "{}{}::{}",
            caps.get(1).map_or("", |m| m.as_str()),
            file_stem(&decode_base32(caps.get(3).unwrap().as_str())),
            caps.get(2).unwrap().as_str()
        )
    });
    demangled
        .split("::")
        .last()
        .unwrap_or(&demangled)
        .to_owned()
}

fn wgsl_to_rust_type(ty: Handle<Type>, ctx: &GlobalCtx) -> String {
    match ctx.type_to_string(ty).as_str() {
        "i32" => "i32",
        "u32" | "atomic<u32>" => "u32",
        "f32" | "atomic<i32>" => "f32",
        "vec2<f32>" => "[f32; 2]",
        "vec2<u32>" => "[u32; 2]",
        "vec2<i32>" => "[i32; 2]",
        "vec3<f32>" | "vec3<u32>" => "[f32; 4]", // vec3 padded to 16 bytes
        "vec4<f32>" => "[f32; 4]",
        "vec4<u32>" => "[u32; 4]",
        "mat4x4<f32>" => "[[f32; 4]; 4]",
        other => panic!("Unsupported WGSL type: {other}"),
    }
    .to_owned()
}

fn wgsl_type_alignment(ty: Handle<Type>, ctx: &GlobalCtx) -> usize {
    match ctx.type_to_string(ty).as_str() {
        "i32" | "u32" | "f32" | "atomic<u32>" | "atomic<i32>" => 4,
        "vec2<f32>" | "vec2<u32>" | "vec2<i32>" => 8,
        "vec3<f32>" | "vec4<f32>" | "mat4x4<f32>" | "vec3<u32>" | "vec4<u32>" => 16,
        other => panic!("Unknown alignment for: {other}"),
    }
}

struct IncludeInfo {
    source: String,
    file_path: String,
    module_name: String,
}

fn create_composer(includes: &[IncludeInfo]) -> Result<Composer, String> {
    let mut composer = Composer::default().with_capabilities(Capabilities::all());
    for inc in includes {
        if let Err(e) = composer.add_composable_module(ComposableModuleDescriptor {
            source: &inc.source,
            file_path: &inc.file_path,
            as_name: Some(inc.module_name.clone()),
            ..Default::default()
        }) {
            return Err(format!(
                "Failed to add include '{}': {}",
                inc.file_path,
                e.emit_to_string(&composer)
            ));
        }
    }
    Ok(composer)
}

fn compile_to_wgsl(module: &naga::Module) -> String {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::empty(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .expect("Shader validation failed");

    naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
        .expect("WGSL generation failed")
}

struct ExtractedType {
    name: String,
    alignment: usize,
    fields: Vec<(String, String)>,
}

struct ExtractedConst {
    name: String,
    ty: &'static str,
    value: String,
}

struct ShaderInfo {
    workgroup_size: [u32; 3],
    types: Vec<ExtractedType>,
    constants: Vec<ExtractedConst>,
    variants: Vec<(String, String)>, // (suffix, wgsl_source)
}

fn extract_shader_info(
    source: &str,
    source_path: &str,
    includes: &[IncludeInfo],
    defines: &[String],
) -> Result<ShaderInfo, String> {
    let mut composer = create_composer(includes)?;
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path: source_path,
            ..Default::default()
        })
        .map_err(|e| e.emit_to_string(&composer))?;

    if module.entry_points.len() != 1 {
        return Err(format!(
            "Expected exactly 1 entry point in '{}', found {}",
            source_path,
            module.entry_points.len()
        ));
    }
    let workgroup_size = module.entry_points[0].workgroup_size;
    let ctx = module.to_ctx();

    // Extract constants
    let constants: Vec<_> = module
        .constants
        .iter()
        .filter_map(|(_, c)| {
            let (ty, value) = match module.global_expressions[c.init] {
                naga::Expression::Literal(lit) => match lit {
                    naga::Literal::F32(v) => ("f32", format!("{v}f32")),
                    naga::Literal::U32(v) => ("u32", format!("{v}u32")),
                    naga::Literal::I32(v) => ("i32", format!("{v}i32")),
                    naga::Literal::Bool(v) => ("bool", format!("{v}")),
                    naga::Literal::F64(v) => ("f64", format!("{v}f64")),
                    naga::Literal::I64(v) => ("i64", format!("{v}i64")),
                    naga::Literal::U64(v) => ("u64", format!("{v}u64")),
                    _ => return None,
                },
                _ => return None,
            };
            Some(ExtractedConst {
                name: demangle_name(c.name.as_ref()?),
                ty,
                value,
            })
        })
        .collect();

    // Extract struct types
    let types: Vec<_> = module
        .types
        .iter()
        .filter_map(|(_, ty)| {
            let naga::TypeInner::Struct { members, .. } = &ty.inner else {
                return None;
            };
            if members.is_empty() {
                return None;
            }
            let name = ty.name.as_ref()?;
            if name.contains("__atomic_compare_exchange_result") {
                return None;
            }
            Some(ExtractedType {
                name: demangle_name(name),
                alignment: members
                    .iter()
                    .map(|m| wgsl_type_alignment(m.ty, &ctx))
                    .max()
                    .unwrap(),
                fields: members
                    .iter()
                    .map(|m| (m.name.clone().unwrap(), wgsl_to_rust_type(m.ty, &ctx)))
                    .collect(),
            })
        })
        .collect();

    // Compile all define combinations
    let mut variants = Vec::new();
    for combo in generate_define_combinations(defines) {
        let suffix = variant_suffix(defines, &combo);
        let mut variant_composer = create_composer(includes)?;
        let module = variant_composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: source_path,
                shader_defs: combo,
                ..Default::default()
            })
            .map_err(|e| e.emit_to_string(&variant_composer))?;
        variants.push((suffix, compile_to_wgsl(&module)));
    }

    Ok(ShaderInfo {
        workgroup_size,
        types,
        constants,
        variants,
    })
}

fn generate_define_combinations(defines: &[String]) -> Vec<HashMap<String, ShaderDefValue>> {
    (0..1usize << defines.len())
        .map(|i| {
            defines
                .iter()
                .enumerate()
                .filter(|(j, _)| (i >> j) & 1 == 1)
                .map(|(_, d)| (d.clone(), ShaderDefValue::Bool(true)))
                .collect()
        })
        .collect()
}

fn variant_suffix(defines: &[String], enabled: &HashMap<String, ShaderDefValue>) -> String {
    let mut active: Vec<_> = defines
        .iter()
        .filter(|d| enabled.contains_key(*d))
        .collect();
    active.sort();
    if active.is_empty() {
        String::new()
    } else {
        format!(
            "_{}",
            active
                .iter()
                .map(|s| s.to_lowercase())
                .collect::<Vec<_>>()
                .join("_")
        )
    }
}

fn to_snake_case(s: &str) -> String {
    s.chars()
        .enumerate()
        .flat_map(|(i, c)| {
            if c.is_uppercase() && i > 0 {
                vec!['_', c.to_ascii_lowercase()]
            } else {
                vec![c.to_ascii_lowercase()]
            }
        })
        .collect()
}

fn generate_code(
    struct_name: &syn::Ident,
    vis: &syn::Visibility,
    defines: &[String],
    info: &ShaderInfo,
    source_path: &str,
    include_paths: &[String],
) -> TokenStream2 {
    let mod_ident = format_ident!("{}", to_snake_case(&struct_name.to_string()));
    let struct_name_str = struct_name.to_string();
    let [wg_x, wg_y, wg_z] = info.workgroup_size;

    let field_idents: Vec<_> = defines
        .iter()
        .map(|d| format_ident!("{}", d.to_lowercase()))
        .collect();

    // Determine bytemuck path based on which crate we're in
    // brush-kernel re-exports bytemuck, but when compiling brush-kernel itself we use crate::
    let is_brush_kernel = std::env::var("CARGO_PKG_NAME")
        .map(|name| name == "brush-kernel")
        .unwrap_or(false);
    let bytemuck_path: TokenStream2 = if is_brush_kernel {
        quote! { crate::bytemuck }
    } else {
        quote! { ::brush_kernel::bytemuck }
    };

    // Type definitions
    let type_defs = info.types.iter().map(|t| {
        let name = format_ident!("{}", t.name);
        let align = proc_macro2::Literal::usize_unsuffixed(t.alignment);
        let bytemuck = &bytemuck_path;
        let fields = t.fields.iter().map(|(fname, ftype)| {
            let fname = format_ident!("{}", fname);
            let ftype: TokenStream2 = ftype.parse().unwrap();
            quote! { pub #fname: #ftype }
        });
        quote! {
            #[repr(C, align(#align))]
            #[derive(Debug, Clone, Copy, #bytemuck::NoUninit)]
            pub struct #name { #(#fields),* }
        }
    });

    // Constant definitions
    let const_defs = info.constants.iter().map(|c| {
        let name = format_ident!("{}", c.name);
        let ty: TokenStream2 = c.ty.parse().unwrap();
        let value: TokenStream2 = c.value.parse().unwrap();
        quote! { pub const #name: #ty = #value; }
    });

    // Shader source constants
    let shader_consts = info.variants.iter().map(|(suffix, wgsl)| {
        let name = format_ident!("SHADER_SOURCE{}", suffix.to_uppercase());
        quote! { const #name: &str = #wgsl; }
    });

    // get_shader_source function
    let get_shader_source = if defines.is_empty() {
        quote! {
            pub fn get_shader_source() -> &'static str { SHADER_SOURCE }
        }
    } else {
        let match_arms = generate_define_combinations(defines)
            .iter()
            .map(|combo| {
                let suffix = variant_suffix(defines, combo);
                let const_name = format_ident!("SHADER_SOURCE{}", suffix.to_uppercase());
                let pattern: Vec<_> = defines.iter().map(|d| combo.contains_key(d)).collect();
                quote! { (#(#pattern),*) => #const_name }
            })
            .collect::<Vec<_>>();

        let params = field_idents.iter().map(|f| quote! { #f: bool });
        quote! {
            pub fn get_shader_source(#(#params),*) -> &'static str {
                match (#(#field_idents),*) { #(#match_arms),* }
            }
        }
    };

    // Struct definition
    let struct_def = if defines.is_empty() {
        quote! { #[derive(Debug, Copy, Clone)] pub struct #struct_name; }
    } else {
        let fields = field_idents.iter().map(|f| quote! { pub #f: bool });
        quote! { #[derive(Debug, Copy, Clone)] pub struct #struct_name { #(#fields),* } }
    };

    // task() constructor
    let task_impl = if defines.is_empty() {
        quote! { pub fn task() -> Box<Self> { Box::new(Self) } }
    } else {
        let params = field_idents.iter().map(|f| quote! { #f: bool });
        quote! {
            #[allow(clippy::too_many_arguments)]
            pub fn task(#(#params),*) -> Box<Self> { Box::new(Self { #(#field_idents),* }) }
        }
    };

    // File tracking for rebuilds
    let track_files = std::iter::once(source_path.to_owned())
        .chain(include_paths.iter().cloned())
        .map(|p| quote! { const _: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #p)); });

    // CubeTask compile implementation
    let get_source_call = if defines.is_empty() {
        quote! { get_shader_source() }
    } else {
        quote! { get_shader_source(#(self.#field_idents),*) }
    };

    quote! {
        #vis mod #mod_ident {
            #(#track_files)*
            #(#type_defs)*
            #(#shader_consts)*
            #get_shader_source
            #struct_def

            impl #struct_name {
                pub const WORKGROUP_SIZE: [u32; 3] = [#wg_x, #wg_y, #wg_z];
                #(#const_defs)*
                #task_impl
            }

            impl<C: burn_cubecl::cubecl::Compiler> burn_cubecl::cubecl::CubeTask<C> for #struct_name {
                fn compile(
                    &self, _: &mut C, _: &C::CompilationOptions, _: burn_cubecl::cubecl::prelude::ExecutionMode,
                ) -> Result<burn_cubecl::cubecl::prelude::CompiledKernel<C>, burn_cubecl::cubecl::CompilationError> {
                    let source = #get_source_call;

                    #[cfg(target_family = "wasm")]
                    let source = if ["subgroupAdd", "subgroupAny", "subgroupMax", "subgroupBroadcast", "subgroupShuffle"]
                        .iter().any(|s| source.contains(s))
                    {
                        format!("enable subgroups;\n{}", source)
                    } else {
                        source.to_string()
                    };

                    Ok(burn_cubecl::cubecl::prelude::CompiledKernel {
                        entrypoint_name: "main".to_owned(),
                        debug_name: Some(#struct_name_str),
                        source: source.to_owned(),
                        repr: None,
                        cube_dim: burn_cubecl::cubecl::CubeDim::new_3d(
                            Self::WORKGROUP_SIZE[0], Self::WORKGROUP_SIZE[1], Self::WORKGROUP_SIZE[2]
                        ),
                        debug_info: None,
                    })
                }
            }

            impl burn_cubecl::kernel::KernelMetadata for #struct_name {
                fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
                    burn_cubecl::cubecl::prelude::KernelId::new::<Self>().info(vec![#(self.#field_idents),*] as Vec<bool>)
                }
            }
        }

        #vis use #mod_ident::#struct_name;
    }
}

#[proc_macro_attribute]
pub fn wgsl_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as WgslKernelArgs);
    let input = parse_macro_input!(item as ItemStruct);

    // Extract defines from bool fields
    let defines: Result<Vec<_>, _> = match &input.fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|f| {
                let is_bool = matches!(&f.ty, syn::Type::Path(p) if p.path.is_ident("bool"));
                if is_bool {
                    Ok(f.ident.as_ref().unwrap().to_string().to_uppercase())
                } else {
                    Err(syn::Error::new(
                        f.ty.span(),
                        "Only bool fields allowed (they become shader defines)",
                    ))
                }
            })
            .collect(),
        Fields::Unit => Ok(vec![]),
        _ => Err(syn::Error::new(
            input.fields.span(),
            "Expected named fields or unit struct",
        )),
    };

    let defines = match defines {
        Ok(d) => d,
        Err(e) => return e.to_compile_error().into(),
    };

    // Read source file
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let source_path = Path::new(&manifest_dir).join(&args.source);
    let source = match std::fs::read_to_string(&source_path) {
        Ok(s) => s,
        Err(e) => {
            return {
                let msg: &str = &format!("Failed to read '{}': {}", source_path.display(), e);
                syn::Error::new(proc_macro2::Span::call_site(), msg)
                    .to_compile_error()
                    .into()
            };
        }
    };

    // Discover and read includes
    let source_dir = source_path.parent().unwrap_or(Path::new(&manifest_dir));
    let explicit_names: HashSet<_> = args.includes.iter().map(|p| file_stem(p)).collect();

    let mut includes: Vec<IncludeInfo> = extract_import_names(&source)
        .into_iter()
        .filter(|name| !explicit_names.contains(name))
        .filter_map(|name| {
            let path = source_dir.join(format!("{name}.wgsl"));
            if !path.exists() {
                return None;
            }
            let source = std::fs::read_to_string(&path).ok()?;
            let file_path = path
                .strip_prefix(&manifest_dir)
                .ok()?
                .to_string_lossy()
                .into_owned();
            Some(IncludeInfo {
                source,
                file_path,
                module_name: name,
            })
        })
        .collect();

    for inc_path in &args.includes {
        let full_path = Path::new(&manifest_dir).join(inc_path);
        let source = match std::fs::read_to_string(&full_path) {
            Ok(s) => s,
            Err(e) => {
                return {
                    let msg: &str = &format!("Failed to read '{}': {}", full_path.display(), e);
                    syn::Error::new(proc_macro2::Span::call_site(), msg)
                        .to_compile_error()
                        .into()
                };
            }
        };
        includes.push(IncludeInfo {
            source,
            file_path: inc_path.clone(),
            module_name: file_stem(inc_path),
        });
    }

    let info = match extract_shader_info(&source, &args.source, &includes, &defines) {
        Ok(info) => info,
        Err(e) => {
            return syn::Error::new(proc_macro2::Span::call_site(), e)
                .to_compile_error()
                .into();
        }
    };
    let include_paths: Vec<_> = includes.iter().map(|i| i.file_path.clone()).collect();

    generate_code(
        &input.ident,
        &input.vis,
        &defines,
        &info,
        &args.source,
        &include_paths,
    )
    .into()
}
