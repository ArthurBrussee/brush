use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use image::{DynamicImage, Rgb32FImage, Rgba32FImage};

use crate::scene::{SceneView, ViewImageType};

// Converts an image to a train sample. The tensor will be a floating point image with a [0, 1] image.
//
// This assume the input image has un-premultiplied alpha, whereas the output has pre-multiplied alpha.
pub fn view_to_sample<B: Backend>(view: &SceneView, device: &B::Device) -> Tensor<B, 3> {
    let image = &view.image;
    let (w, h) = (image.width(), image.height());

    let tensor_data = if image.color().has_alpha() {
        // Assume image has un-multiplied alpha and convert it to pre-multiplied.
        let mut rgba = image.to_rgba32f();
        if view.img_type == ViewImageType::Alpha {
            for pixel in rgba.pixels_mut() {
                let a = pixel[3];
                pixel[0] *= a;
                pixel[1] *= a;
                pixel[2] *= a;
            }
        }
        TensorData::new(rgba.into_vec(), [h as usize, w as usize, 4])
    } else {
        TensorData::new(image.to_rgb32f().into_vec(), [h as usize, w as usize, 3])
    };

    Tensor::from_data(tensor_data, device)
}

pub trait TensorDataToImage {
    fn into_image(self) -> DynamicImage;
}

pub fn tensor_into_image(data: TensorData) -> DynamicImage {
    let [h, w, c] = [data.shape[0], data.shape[1], data.shape[2]];

    let img: DynamicImage = match data.dtype {
        DType::F32 => {
            let data = data.into_vec::<f32>().expect("Wrong type");
            if c == 3 {
                Rgb32FImage::from_raw(w as u32, h as u32, data)
                    .expect("Failed to create image from tensor")
                    .into()
            } else if c == 4 {
                Rgba32FImage::from_raw(w as u32, h as u32, data)
                    .expect("Failed to create image from tensor")
                    .into()
            } else {
                panic!("Unsupported number of channels: {c}");
            }
        }
        _ => panic!("unsupported dtype {:?}", data.dtype),
    };

    img
}
