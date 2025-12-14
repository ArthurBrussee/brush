//! Procedural macro for generating WGSL kernel wrappers.
//!
//! # Basic Usage
//!
//! Imports from the same directory are auto-discovered from `#import` statements
//! in the WGSL source, so you typically only need to specify the source file:
//!
//! ```ignore
//! #[wgsl_kernel(source = "src/shaders/rasterize.wgsl")]
//! pub struct Rasterize {
//!     pub bwd_info: bool,
//!     pub webgpu: bool,
//! }
//! ```
//!
//! For imports from other crates, use explicit `includes`:
//!
//! ```ignore
//! #[wgsl_kernel(
//!     source = "src/shaders/project_backwards.wgsl",
//!     includes = ["../brush-render/src/shaders/helpers.wgsl"],
//! )]
//! pub struct ProjectBackwards;
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

use naga::{Handle, Type, proc::GlobalCtx, valid::Capabilities};
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
use wgpu::naga::{self, common::wgsl::TypeContext};

const DECORATION_PRE: &str = "X_naga_oil_mod_X";
const DECORATION_POST: &str = "X";

struct WgslKernelArgs {
    source: String,
    includes: Vec<String>,
}

impl Parse for WgslKernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut source = None;
        let mut includes = Vec::new();

        let metas = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;

        for meta in metas {
            match meta {
                Meta::NameValue(nv) if nv.path.is_ident("source") => {
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &nv.value
                    {
                        source = Some(s.value());
                    } else {
                        return Err(syn::Error::new(nv.value.span(), "expected string literal"));
                    }
                }
                Meta::NameValue(nv) if nv.path.is_ident("includes") => {
                    // Parse as array: includes = ["a.wgsl", "b.wgsl"]
                    if let Expr::Array(arr) = &nv.value {
                        for elem in &arr.elems {
                            if let Expr::Lit(ExprLit {
                                lit: Lit::Str(s), ..
                            }) = elem
                            {
                                includes.push(s.value());
                            } else {
                                return Err(syn::Error::new(
                                    elem.span(),
                                    "expected string literal",
                                ));
                            }
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
                        "unknown attribute, expected `source` or `includes`",
                    ));
                }
            }
        }

        let source = source.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `source` attribute")
        })?;

        Ok(WgslKernelArgs { source, includes })
    }
}

/// Extract import names from WGSL source.
/// Matches patterns like `#import helpers`, `#import helpers;`, `#import prefix_sum_helpers as helpers`
fn extract_import_names(source: &str) -> Vec<String> {
    static IMPORT_REGEX: OnceLock<Regex> = OnceLock::new();
    let re = IMPORT_REGEX.get_or_init(|| {
        // Match #import <name> with optional `as <alias>` and optional semicolon
        Regex::new(r"#import\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?;?")
            .unwrap()
    });

    re.captures_iter(source)
        .map(|cap| cap[1].to_string())
        .collect()
}

fn make_valid_rust_import(value: &str) -> String {
    let v = value.replace("\"../", "").replace('"', "");
    std::path::Path::new(&v)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or(&v)
        .to_owned()
}

fn decode(from: &str) -> String {
    String::from_utf8(data_encoding::BASE32_NOPAD.decode(from.as_bytes()).unwrap()).unwrap()
}

fn undecorate_regex() -> &'static Regex {
    static MEM: OnceLock<Regex> = OnceLock::new();

    MEM.get_or_init(|| {
        Regex::new(
            format!(
                r"(\x1B\[\d+\w)?([\w\d_]+){}([A-Z0-9]*){}",
                regex_syntax::escape(DECORATION_PRE),
                regex_syntax::escape(DECORATION_POST)
            )
            .as_str(),
        )
        .unwrap()
    })
}

fn demangle_str(string: &str) -> std::borrow::Cow<'_, str> {
    undecorate_regex().replace_all(string, |caps: &regex::Captures| {
        format!(
            "{}{}::{}",
            caps.get(1).map_or("", |cc| cc.as_str()),
            make_valid_rust_import(&decode(caps.get(3).unwrap().as_str())),
            caps.get(2).unwrap().as_str()
        )
    })
}

fn mod_name_from_mangled(string: &str) -> (String, String) {
    let demangled = demangle_str(string);
    let mut parts = demangled.as_ref().split("::").collect::<Vec<&str>>();
    let name = parts.pop().unwrap().to_owned();
    let mod_name = parts.join("::");
    (mod_name, name)
}

fn rust_type_name(ty: Handle<naga::Type>, ctx: &GlobalCtx) -> String {
    let wgsl_name = ctx.type_to_string(ty);

    match wgsl_name.as_str() {
        "i32" | "u32" | "f32" => wgsl_name,
        "atomic<u32>" => "u32".to_owned(),
        "atomic<i32>" => "i32".to_owned(),
        "vec2<f32>" => "[f32; 2]".to_owned(),
        "vec4<f32>" => "[f32; 4]".to_owned(),
        "mat4x4<f32>" => "[[f32; 4]; 4]".to_owned(),
        "vec2<u32>" => "[u32; 2]".to_owned(),
        "vec2<i32>" => "[i32; 2]".to_owned(),
        "vec3<u32>" => "[u32; 4]".to_owned(),
        "vec3<f32>" => "[f32; 4]".to_owned(),
        "vec4<u32>" => "[u32; 4]".to_owned(),
        _ => panic!("Unsupported WGSL type: {}", wgsl_name),
    }
}

fn alignment_of(ty: Handle<Type>, ctx: &GlobalCtx) -> usize {
    let wgsl_name = ctx.type_to_string(ty);

    match wgsl_name.as_str() {
        "i32" | "u32" | "f32" | "atomic<u32>" | "atomic<i32>" => 4,
        "vec2<f32>" | "vec2<u32>" | "vec2<i32>" => 8,
        "vec3<f32>" | "vec4<f32>" | "mat4x4<f32>" | "vec4<u32>" => 16,
        _ => panic!("Unknown alignment for type: {}", wgsl_name),
    }
}

struct IncludeInfo {
    source: String,
    file_path: String,
    as_name: String,
}

fn create_composer_with_includes(includes: &[IncludeInfo]) -> Composer {
    let mut composer = Composer::default().with_capabilities(Capabilities::all());

    for include in includes {
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &include.source,
                file_path: &include.file_path,
                as_name: Some(include.as_name.clone()),
                ..Default::default()
            })
            .expect("Failed to add composable module");
    }

    composer
}

fn compile_to_wgsl(module: &naga::Module) -> String {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::empty(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .expect("Failed to validate module");

    naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
        .expect("Failed to convert naga module to WGSL")
}

fn generate_define_combinations(defines: &[String]) -> Vec<HashMap<String, ShaderDefValue>> {
    let n = defines.len();
    let count = 1usize << n;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let mut map = HashMap::new();
        for (j, define) in defines.iter().enumerate() {
            if (i >> j) & 1 == 1 {
                map.insert(define.clone(), ShaderDefValue::Bool(true));
            }
        }
        result.push(map);
    }

    result
}

fn variant_suffix(defines: &[String], enabled: &HashMap<String, ShaderDefValue>) -> String {
    let mut parts: Vec<&str> = defines
        .iter()
        .filter(|d| enabled.contains_key(*d))
        .map(|d| d.as_str())
        .collect();
    parts.sort();

    if parts.is_empty() {
        String::new()
    } else {
        format!("_{}", parts.join("_").to_lowercase())
    }
}

struct ExtractedType {
    name: String,
    alignment: usize,
    fields: Vec<(String, String)>, // (field_name, rust_type)
}

struct ExtractedConstant {
    name: String,
    rust_type: String,
    value: String,
}

struct ShaderInfo {
    workgroup_size: [u32; 3],
    types: Vec<ExtractedType>,
    constants: Vec<ExtractedConstant>,
    variants: Vec<(String, String)>, // (suffix, compiled_wgsl)
}

fn extract_shader_info(
    source: &str,
    source_path: &str,
    includes: &[IncludeInfo],
    defines: &[String],
) -> ShaderInfo {
    // Compile with no defines first to extract metadata
    let mut composer = create_composer_with_includes(includes);
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path: source_path,
            ..Default::default()
        })
        .expect("Failed to compile shader");

    let entries = &module.entry_points;
    assert!(
        entries.len() == 1,
        "Must have exactly 1 entry point per shader file"
    );
    let entry = &entries[0];
    let workgroup_size = entry.workgroup_size;
    let ctx = &module.to_ctx();

    // Extract constants
    let mut constants = Vec::new();
    for (_, constant) in module.constants.iter() {
        let type_and_value = match module.global_expressions[constant.init] {
            naga::Expression::Literal(literal) => match literal {
                naga::Literal::F64(v) => Some(("f64", format!("{v}f64"))),
                naga::Literal::F32(v) => Some(("f32", format!("{v}f32"))),
                naga::Literal::U32(v) => Some(("u32", format!("{v}u32"))),
                naga::Literal::I32(v) => Some(("i32", format!("{v}i32"))),
                naga::Literal::Bool(v) => Some(("bool", format!("{v}"))),
                naga::Literal::I64(v) => Some(("i64", format!("{v}i64"))),
                naga::Literal::U64(v) => Some(("u64", format!("{v}u64"))),
                naga::Literal::AbstractInt(v) => Some(("i64", format!("{v}i64"))),
                naga::Literal::AbstractFloat(v) => Some(("f64", format!("{v}f64"))),
                naga::Literal::F16(_) => None, // Skip f16 for now
            },
            _ => None,
        };

        if let Some((rust_type, value)) = type_and_value
            && let Some(mangled_name) = constant.name.as_ref()
        {
            let (_, name) = mod_name_from_mangled(mangled_name);
            constants.push(ExtractedConstant {
                name,
                rust_type: rust_type.to_string(),
                value,
            });
        }
    }

    let mut types = Vec::new();
    for (_, ty) in module.types.iter() {
        if let naga::TypeInner::Struct { members, span: _ } = &ty.inner {
            if members.is_empty() {
                continue;
            }

            if let Some(mangled_name) = ty.name.as_ref() {
                // Skip builtins
                if mangled_name.contains("__atomic_compare_exchange_result") {
                    continue;
                }

                let (_, name) = mod_name_from_mangled(mangled_name);

                let max_align = members
                    .iter()
                    .map(|x| alignment_of(x.ty, ctx))
                    .max()
                    .unwrap();

                let fields: Vec<(String, String)> = members
                    .iter()
                    .map(|m| (m.name.as_ref().unwrap().clone(), rust_type_name(m.ty, ctx)))
                    .collect();

                types.push(ExtractedType {
                    name,
                    alignment: max_align,
                    fields,
                });
            }
        }
    }

    // Compile all define combinations
    let combinations = generate_define_combinations(defines);
    let mut variants = Vec::new();

    for combo in &combinations {
        let suffix = variant_suffix(defines, combo);
        let mut variant_composer = create_composer_with_includes(includes);
        let variant_module = variant_composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: source_path,
                shader_defs: combo.clone(),
                ..Default::default()
            })
            .expect("Failed to compile shader variant");

        let wgsl_output = compile_to_wgsl(&variant_module);
        variants.push((suffix, wgsl_output));
    }

    ShaderInfo {
        workgroup_size,
        types,
        constants,
        variants,
    }
}

/// Convert a PascalCase identifier to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

fn generate_code(
    struct_name: &syn::Ident,
    struct_vis: &syn::Visibility,
    defines: &[String],
    info: &ShaderInfo,
    source_path: &str,
    include_paths: Vec<String>,
) -> TokenStream2 {
    // Auto-detect crate path: use `crate` if we're inside brush-kernel, otherwise `brush_kernel`
    let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let crate_path = if pkg_name == "brush-kernel" {
        "crate"
    } else {
        "brush_kernel"
    };
    let crate_path_tokens: TokenStream2 = crate_path.parse().unwrap();
    let [wg_x, wg_y, wg_z] = info.workgroup_size;

    // Field names for the compile method call
    let field_names_for_call: Vec<_> = defines
        .iter()
        .map(|d| format_ident!("{}", d.to_lowercase()))
        .collect();

    // Generate type definitions
    let type_defs: Vec<TokenStream2> = info
        .types
        .iter()
        .map(|t| {
            let name = format_ident!("{}", t.name);
            let align = proc_macro2::Literal::usize_unsuffixed(t.alignment);
            let fields: Vec<TokenStream2> = t
                .fields
                .iter()
                .map(|(fname, ftype)| {
                    let fname = format_ident!("{}", fname);
                    let ftype: TokenStream2 = ftype.parse().unwrap();
                    quote! { pub #fname: #ftype }
                })
                .collect();

            quote! {
                #[repr(C, align(#align))]
                #[derive(bytemuck::Pod, bytemuck::Zeroable, Debug, Clone, Copy)]
                pub struct #name {
                    #(#fields),*
                }
            }
        })
        .collect();

    // Generate constant definitions
    let const_defs: Vec<TokenStream2> = info
        .constants
        .iter()
        .map(|c| {
            let name = format_ident!("{}", c.name);
            let ty: TokenStream2 = c.rust_type.parse().unwrap();
            let value: TokenStream2 = c.value.parse().unwrap();
            quote! { pub const #name: #ty = #value; }
        })
        .collect();

    // Generate shader source constants
    let shader_consts: Vec<TokenStream2> = info
        .variants
        .iter()
        .map(|(suffix, wgsl)| {
            let const_name = if suffix.is_empty() {
                format_ident!("SHADER_SOURCE")
            } else {
                format_ident!("SHADER_SOURCE{}", suffix.to_uppercase())
            };
            quote! { const #const_name: &str = #wgsl; }
        })
        .collect();

    // Generate get_shader_source function
    let get_shader_source = if defines.is_empty() {
        quote! {
            pub fn get_shader_source() -> &'static str {
                SHADER_SOURCE
            }
        }
    } else {
        let params: Vec<TokenStream2> = defines
            .iter()
            .map(|d| {
                let name = format_ident!("{}", d.to_lowercase());
                quote! { #name: bool }
            })
            .collect();

        let combinations = generate_define_combinations(defines);
        let match_arms: Vec<TokenStream2> = combinations
            .iter()
            .map(|combo| {
                let suffix = variant_suffix(defines, combo);
                let const_name = if suffix.is_empty() {
                    format_ident!("SHADER_SOURCE")
                } else {
                    format_ident!("SHADER_SOURCE{}", suffix.to_uppercase())
                };

                let pattern_parts: Vec<TokenStream2> = defines
                    .iter()
                    .map(|d| {
                        if combo.contains_key(d) {
                            quote! { true }
                        } else {
                            quote! { false }
                        }
                    })
                    .collect();

                let pattern = if pattern_parts.len() == 1 {
                    quote! { #(#pattern_parts),* }
                } else {
                    quote! { (#(#pattern_parts),*) }
                };

                quote! { #pattern => #const_name }
            })
            .collect();

        let match_expr = if defines.len() == 1 {
            let name = format_ident!("{}", defines[0].to_lowercase());
            quote! { #name }
        } else {
            let names: Vec<_> = defines
                .iter()
                .map(|d| format_ident!("{}", d.to_lowercase()))
                .collect();
            quote! { (#(#names),*) }
        };

        quote! {
            pub fn get_shader_source(#(#params),*) -> &'static str {
                match #match_expr {
                    #(#match_arms),*
                }
            }
        }
    };

    // Generate struct fields
    let struct_fields: Vec<TokenStream2> = defines
        .iter()
        .map(|d| {
            let name = format_ident!("{}", d.to_lowercase());
            quote! { pub #name: bool }
        })
        .collect();

    // Generate struct definition - always pub inside the module since module visibility controls access
    let struct_def = if defines.is_empty() {
        quote! {
            #[derive(Debug, Copy, Clone)]
            pub struct #struct_name;
        }
    } else {
        quote! {
            #[derive(Debug, Copy, Clone)]
            pub struct #struct_name {
                #(#struct_fields),*
            }
        }
    };

    // Generate task() constructor
    let task_impl = if defines.is_empty() {
        quote! {
            pub fn task() -> Box<Self> {
                Box::new(Self)
            }
        }
    } else {
        let params: Vec<TokenStream2> = defines
            .iter()
            .map(|d| {
                let name = format_ident!("{}", d.to_lowercase());
                quote! { #name: bool }
            })
            .collect();
        let field_names: Vec<_> = defines
            .iter()
            .map(|d| format_ident!("{}", d.to_lowercase()))
            .collect();

        quote! {
            #[allow(clippy::too_many_arguments)]
            pub fn task(#(#params),*) -> Box<Self> {
                Box::new(Self { #(#field_names),* })
            }
        }
    };

    let struct_name_str = struct_name.to_string();

    // Generate KernelMetadata impl
    let kernel_id_args = if defines.is_empty() {
        quote! { &[] }
    } else {
        let field_names: Vec<_> = defines
            .iter()
            .map(|d| format_ident!("{}", d.to_lowercase()))
            .collect();
        quote! { &[#(self.#field_names),*] }
    };

    // Generate file tracking (ensures cargo rebuilds on file changes)
    // Use concat! with CARGO_MANIFEST_DIR to get absolute paths
    let track_source = quote! {
        const _: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path));
    };
    let track_includes: Vec<TokenStream2> = include_paths
        .into_iter()
        .map(|p| {
            quote! { const _: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #p)); }
        })
        .collect();

    // Generate module name from struct name (convert to snake_case)
    let mod_name = to_snake_case(&struct_name.to_string());
    let mod_ident = format_ident!("{}", mod_name);

    quote! {
        #struct_vis mod #mod_ident {
            // Track source files for rebuilds
            #track_source
            #(#track_includes)*

            // Extracted types
            #(#type_defs)*

            // Compiled shader variants
            #(#shader_consts)*

            // Shader source accessor
            #get_shader_source

            // Kernel struct
            #struct_def

            impl #struct_name {
                /// Workgroup size for this kernel
                pub const WORKGROUP_SIZE: [u32; 3] = [#wg_x, #wg_y, #wg_z];

                // Extracted constants from shader
                #(#const_defs)*

                #task_impl
            }

            impl<C: #crate_path_tokens::Compiler> #crate_path_tokens::CubeTask<C> for #struct_name {
                fn compile(
                    &self,
                    _compiler: &mut C,
                    _compilation_options: &C::CompilationOptions,
                    _mode: #crate_path_tokens::ExecutionMode,
                ) -> Result<#crate_path_tokens::CompiledKernel<C>, #crate_path_tokens::CompilationError> {
                    let source = get_shader_source(#(self.#field_names_for_call),*);
                    let module = #crate_path_tokens::parse_wgsl(source);
                    #crate_path_tokens::module_to_compiled(#struct_name_str, &module, Self::WORKGROUP_SIZE)
                }
            }

            impl #crate_path_tokens::KernelMetadata for #struct_name {
                fn id(&self) -> #crate_path_tokens::KernelId {
                    #crate_path_tokens::calc_kernel_id::<Self>(#kernel_id_args)
                }
            }
        }

        // Re-export struct from module
        #struct_vis use #mod_ident::#struct_name;
    }
}

// ============================================================================
// The proc macro
// ============================================================================

/// Attribute macro for generating WGSL kernel wrappers.
///
/// # Arguments
/// - `source`: Path to the WGSL source file (required)
/// - `includes`: Array of paths to include files (optional)
///
/// The crate path for `brush_kernel` is auto-detected based on `CARGO_PKG_NAME`.
///
/// # Example
/// ```ignore
/// #[wgsl_kernel(
///     source = "src/shaders/rasterize.wgsl",
///     includes = ["src/shaders/helpers.wgsl"],
/// )]
/// pub struct Rasterize {
///     pub bwd_info: bool,
///     pub webgpu: bool,
/// }
/// ```
///
/// The bool fields become shader defines. A struct with no fields creates a shader
/// with no defines.
#[proc_macro_attribute]
pub fn wgsl_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as WgslKernelArgs);
    let input = parse_macro_input!(item as ItemStruct);

    // Extract defines from struct fields (must all be bool)
    let mut defines = Vec::new();
    if let Fields::Named(fields) = &input.fields {
        for field in &fields.named {
            let field_name = field.ident.as_ref().unwrap().to_string();

            // Check that field type is bool
            if let syn::Type::Path(type_path) = &field.ty {
                if type_path.path.is_ident("bool") {
                    defines.push(field_name.to_uppercase());
                } else {
                    return syn::Error::new(
                        field.ty.span(),
                        "Only bool fields are supported (they become shader defines)",
                    )
                    .to_compile_error()
                    .into();
                }
            } else {
                return syn::Error::new(
                    field.ty.span(),
                    "Only bool fields are supported (they become shader defines)",
                )
                .to_compile_error()
                .into();
            }
        }
    } else if !matches!(input.fields, Fields::Unit) {
        return syn::Error::new(
            input.fields.span(),
            "Expected named fields (bool defines) or unit struct",
        )
        .to_compile_error()
        .into();
    }

    // Read source files
    // Note: We need to resolve paths relative to the crate root
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    let source_path = std::path::Path::new(&manifest_dir).join(&args.source);
    let source = match std::fs::read_to_string(&source_path) {
        Ok(s) => s,
        Err(e) => {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "Failed to read source file '{}': {}",
                    source_path.display(),
                    e
                ),
            )
            .to_compile_error()
            .into();
        }
    };

    // Auto-discover imports from the source file
    let import_names = extract_import_names(&source);
    let source_dir = std::path::Path::new(&manifest_dir)
        .join(&args.source)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from(&manifest_dir));

    // Build a set of explicitly specified include names (by their module name)
    let explicit_include_names: std::collections::HashSet<String> = args
        .includes
        .iter()
        .map(|inc| make_valid_rust_import(inc))
        .collect();

    // Read include files - first auto-discovered ones, then explicit ones
    let mut include_infos = Vec::new();

    // Auto-discover imports from the same directory
    for import_name in &import_names {
        // Skip if explicitly specified (explicit includes take precedence)
        if explicit_include_names.contains(import_name) {
            continue;
        }

        // Look for <import_name>.wgsl in the same directory as the source
        let import_filename = format!("{}.wgsl", import_name);
        let import_path = source_dir.join(&import_filename);

        if import_path.exists() {
            let include_source = match std::fs::read_to_string(&import_path) {
                Ok(s) => s,
                Err(e) => {
                    return syn::Error::new(
                        proc_macro2::Span::call_site(),
                        format!(
                            "Failed to read auto-discovered import '{}': {}",
                            import_path.display(),
                            e
                        ),
                    )
                    .to_compile_error()
                    .into();
                }
            };

            // Construct relative path from manifest dir for file_path
            let relative_path = import_path
                .strip_prefix(&manifest_dir)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| import_path.to_string_lossy().to_string());

            include_infos.push(IncludeInfo {
                source: include_source,
                file_path: relative_path,
                as_name: import_name.clone(),
            });
        }
        // If not found in same directory, it might be provided explicitly or will cause a compile error
    }

    // Add explicit includes (these override/supplement auto-discovered ones)
    for include in &args.includes {
        let include_path = std::path::Path::new(&manifest_dir).join(include);
        let include_source = match std::fs::read_to_string(&include_path) {
            Ok(s) => s,
            Err(e) => {
                return syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!(
                        "Failed to read include file '{}': {}",
                        include_path.display(),
                        e
                    ),
                )
                .to_compile_error()
                .into();
            }
        };
        let as_name = make_valid_rust_import(include);
        include_infos.push(IncludeInfo {
            source: include_source,
            file_path: include.clone(),
            as_name,
        });
    }

    // Extract shader info
    let info = extract_shader_info(&source, &args.source, &include_infos, &defines);

    // Collect all include paths for file tracking (auto-discovered + explicit)
    let all_include_paths: Vec<String> =
        include_infos.iter().map(|i| i.file_path.clone()).collect();

    // Generate code
    let generated = generate_code(
        &input.ident,
        &input.vis,
        &defines,
        &info,
        &args.source,
        all_include_paths,
    );

    generated.into()
}
