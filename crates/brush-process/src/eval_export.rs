use anyhow::Result;
use brush_train::eval::EvalSample;
use burn::prelude::Backend;
use std::path::Path;

#[allow(unused)]
pub async fn eval_save_to_disk<B: Backend>(sample: &EvalSample<B>, path: &Path) -> Result<()> {
    // TODO: Maybe figure out how to do this on WASM.
    #[cfg(not(target_family = "wasm"))]
    {
        use image::Rgb32FImage;
        log::info!("Saving eval image to disk.");
        let img = sample.rendered.clone();
        let [h, w, _] = [img.dims()[0], img.dims()[1], img.dims()[2]];
        let data = sample
            .rendered
            .clone()
            .into_data_async()
            .await
            .into_vec::<f32>()
            .expect("Wrong type");
        let img: image::DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
            .expect("Failed to create image from tensor")
            .into();
        let img: image::DynamicImage = img.into_rgb8().into();
        let parent = path.parent().expect("Eval must have a filename");
        tokio::fs::create_dir_all(parent).await?;
        log::info!("Saving eval view to {path:?}");
        img.save(path)?;
    }
    Ok(())
}
