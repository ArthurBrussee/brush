pub mod process;

mod train_stream;
mod view_stream;

use burn::tensor::{DType, TensorData};
use image::{DynamicImage, Rgb32FImage, Rgba32FImage};

pub(crate) fn tensor_into_image(data: TensorData) -> DynamicImage {
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
