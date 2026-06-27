use std::io::Write;
use std::path::Path;

use brush_render::{TextureMode, camera::Camera, gaussian_splats::Splats, render_splats};
use ffmpeg_sidecar::command::FfmpegCommand;
use glam::Vec3;

/// Renders `cameras` against `splats` and encodes them to an H.264 MP4 at
/// `path` with ffmpeg. `on_progress(done, total)` is called after each frame.
#[allow(clippy::too_many_arguments)]
pub async fn render_to_mp4(
    splats: Splats,
    cameras: Vec<Camera>,
    fps: usize,
    background: Vec3,
    splat_scale: Option<f32>,
    width: u32,
    height: u32,
    path: &Path,
    mut on_progress: impl FnMut(usize, usize),
) -> anyhow::Result<()> {
    let mut child = FfmpegCommand::new()
        .args(["-f", "rawvideo", "-pixel_format", "rgb24"])
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
        let (image, _) = render_splats(
            splats.clone(),
            camera,
            img_size,
            background,
            splat_scale,
            TextureMode::Float,
        )
        .await;

        // Float render is [h, w, 4] in 0..1; pack to rgb24, dropping alpha.
        let data = image
            .to_data_async()
            .await
            .map_err(|e| anyhow::anyhow!("frame readback failed: {e:?}"))?;
        let floats = data
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("unexpected frame format: {e:?}"))?;

        let mut rgb = vec![0u8; (width * height * 3) as usize];
        for px in 0..(width * height) as usize {
            for c in 0..3 {
                rgb[px * 3 + c] = (floats[px * 4 + c].clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
        stdin.write_all(&rgb)?;
        on_progress(i + 1, total);
    }

    drop(stdin);
    if !child.wait()?.success() {
        anyhow::bail!("ffmpeg failed to encode the video");
    }
    Ok(())
}
