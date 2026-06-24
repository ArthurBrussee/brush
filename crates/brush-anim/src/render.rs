use std::io::Write;
use std::path::Path;

use brush_render::{TextureMode, camera::Camera, gaussian_splats::Splats, render_splats};
use ffmpeg_sidecar::command::FfmpegCommand;

use crate::config::VideoSettings;

/// Renders `cameras` against `splats` and encodes them to an H.264 MP4 at
/// `path` with ffmpeg. `on_progress(done, total)` is called after each frame.
pub async fn render_to_mp4(
    splats: Splats,
    cameras: &[Camera],
    settings: &VideoSettings,
    path: &Path,
    mut on_progress: impl FnMut(usize, usize),
) -> anyhow::Result<()> {
    let VideoSettings {
        width,
        height,
        fps,
        background,
        splat_scale,
    } = *settings;

    let mut child = FfmpegCommand::new()
        // Packed renders are RGBA8, one u32 per pixel; feed those bytes
        // straight through and let ffmpeg drop alpha on the yuv420p convert.
        .args(["-f", "rawvideo", "-pixel_format", "rgba"])
        .args(["-video_size", &format!("{width}x{height}")])
        .args(["-framerate", &fps.to_string()])
        .args(["-i", "-"])
        .args([
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
        ])
        // Quiet stderr so its pipe can't fill and stall the encoder.
        .args(["-loglevel", "error", "-nostats", "-y"])
        .arg(path)
        .spawn()?;

    let mut stdin = child
        .take_stdin()
        .ok_or_else(|| anyhow::anyhow!("ffmpeg stdin unavailable"))?;

    let img_size = glam::uvec2(width, height);
    let total = cameras.len();
    for (i, camera) in cameras.iter().enumerate() {
        // `Packed` keeps this on the forward-only raster pass and hands back
        // an [h, w, 1] u32 image already quantized to RGBA8 on the GPU — no
        // backward bookkeeping and no CPU conversion, unlike `Float`.
        let (image, _) = render_splats(
            splats.clone(),
            camera,
            img_size,
            background,
            splat_scale,
            TextureMode::Packed,
        )
        .await;

        let data = image
            .to_data_async()
            .await
            .map_err(|e| anyhow::anyhow!("frame readback failed: {e:?}"))?;
        stdin.write_all(data.as_bytes())?;
        on_progress(i + 1, total);
    }

    drop(stdin);
    if !child.wait()?.success() {
        anyhow::bail!("ffmpeg failed to encode the video");
    }
    Ok(())
}
