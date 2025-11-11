use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// The maximum number of SH rest coefficients we support (SH degree 4: 72 coefficients)
const MAX_SH_COEFF_COUNT: usize = 72;

fn sh_field_idents() -> impl Iterator<Item = proc_macro2::Ident> {
    (0..MAX_SH_COEFF_COUNT).map(|i| quote::format_ident!("f_rest_{}", i))
}

/// Generates a macro that expands to the SH field name array at compile time.
/// This allows runtime code to use the same field names without allocation.
#[proc_macro]
pub fn sh_field_names(_input: TokenStream) -> TokenStream {
    let names: Vec<_> = (0..MAX_SH_COEFF_COUNT)
        .map(|i| format!("f_rest_{i}"))
        .collect();
    let expanded = quote! {
        [#(#names),*]
    };
    TokenStream::from(expanded)
}

/// Generates SH rest coefficient fields (f_rest_0 through f_rest_71) for structs.
///
/// The macro looks for a marker field `_sh_rest_fields: ()` and replaces it with
/// the actual 72 f_rest_N fields, preserving any attributes on the marker field.
#[proc_macro_attribute]
pub fn generate_sh_fields(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();

    // Extract the struct fields
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("GenerateShFields only works on structs with named fields"),
        },
        _ => panic!("GenerateShFields only works on structs"),
    };

    // Process fields: find the marker field and replace it with SH fields
    let mut new_fields = vec![];
    let mut found_marker = false;

    for field in fields.iter() {
        if let Some(ident) = &field.ident {
            if ident == "_sh_rest_fields" {
                found_marker = true;
                let attrs = &field.attrs;
                // Generate all SH rest coefficient fields with the same attributes
                for field_name in sh_field_idents() {
                    new_fields.push(quote! {
                        #(#attrs)*
                        pub(crate) #field_name: f32
                    });
                }
            } else {
                // Keep the original field
                let attrs = &field.attrs;
                let vis = &field.vis;
                let ident = &field.ident;
                let ty = &field.ty;
                new_fields.push(quote! {
                    #(#attrs)*
                    #vis #ident: #ty
                });
            }
        }
    }

    if !found_marker {
        panic!("GenerateShFields requires a marker field `_sh_rest_fields: ()`");
    }

    let attrs = &input.attrs;

    let expanded = quote! {
        #(#attrs)*
        pub struct #name #impl_generics #where_clause {
            #(#new_fields),*
        }
    };

    TokenStream::from(expanded)
}

/// Helper macro to generate the sh_rest_coeffs() method that returns all 72 coefficients
#[proc_macro]
pub fn impl_coeffs(input: TokenStream) -> TokenStream {
    let type_name: syn::Ident = parse_macro_input!(input);

    let field_refs: Vec<_> = sh_field_idents()
        .map(|field| quote! { self.#field })
        .collect();

    let count = MAX_SH_COEFF_COUNT;
    let expanded = quote! {
        impl #type_name {
            pub fn sh_rest_coeffs(&self) -> [f32; #count] {
                [#(#field_refs),*]
            }
        }
    };

    TokenStream::from(expanded)
}
