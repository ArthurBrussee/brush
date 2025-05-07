use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::TensorData;
use burn::tensor::{Tensor, backend::Backend};
use burn_lpips::load_vgg_lpips;
use image::{ImageReader, imageops::FilterType};

fn image_to_nchw<B: Backend>(device: &B::Device, img: image::DynamicImage) -> Tensor<B, 4> {
    // Resize to 64x64
    let resized = img.resize_exact(64, 64, FilterType::Lanczos3);
    // Convert to RGB float array
    let rgb_img = resized.to_rgb32f().into_vec();
    let data = TensorData::new(rgb_img, [1, 64, 64, 3]);
    let tens = Tensor::from_data(data, device);
    tens.permute([0, 3, 1, 2])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();

    // Load the LPIPS model
    let model = load_vgg_lpips(&device);

    // Load and preprocess the images
    let image1 = ImageReader::open("./examples/apple.jpg")?.decode()?;
    let image2 = ImageReader::open("./examples/pear.png")?.decode()?;

    let tensor1 = image_to_nchw::<Wgpu>(&device, image1);
    let tensor2 = image_to_nchw::<Wgpu>(&device, image2);

    println!(
        "Converted images to tensors with shape: {:?} and {:?}",
        tensor1.shape(),
        tensor2.shape()
    );

    // Calculate LPIPS similarity score between the two images
    let similarity_score = model.lpips(tensor1, tensor2);

    println!("LPIPS similarity score: {}", similarity_score.into_scalar());

    Ok(())
}
