use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = Vector3, js_namespace = THREE)]
    pub type ThreeVector3;

    #[wasm_bindgen(method, getter)]
    pub fn x(this: &ThreeVector3) -> f64;

    #[wasm_bindgen(method, getter)]
    pub fn y(this: &ThreeVector3) -> f64;

    #[wasm_bindgen(method, getter)]
    pub fn z(this: &ThreeVector3) -> f64;
}

impl ThreeVector3 {
    pub fn to_glam(&self) -> glam::Vec3 {
        glam::Vec3::new(self.x() as f32, self.y() as f32, self.z() as f32)
    }
}
