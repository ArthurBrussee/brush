use std::io::{self};
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncReadExt;
use tokio::io::{AsyncBufRead, AsyncRead};
use tokio_with_wasm::alias as tokio_wasm;

// TODO: Really these should each hold their respective params but bit of an annoying refactor. We just need
// basic params.
#[derive(Debug, Clone)]
pub enum CameraModel {
    SimplePinhole,
    Pinhole,
    SimpleRadial,
    Radial,
    OpenCV,
    OpenCvFishEye,
    FullOpenCV,
    Fov,
    SimpleRadialFisheye,
    RadialFisheye,
    ThinPrismFisheye,
}

impl CameraModel {
    fn from_id(id: i32) -> Option<Self> {
        match id {
            0 => Some(Self::SimplePinhole),
            1 => Some(Self::Pinhole),
            2 => Some(Self::SimpleRadial),
            3 => Some(Self::Radial),
            4 => Some(Self::OpenCV),
            5 => Some(Self::OpenCvFishEye),
            6 => Some(Self::FullOpenCV),
            7 => Some(Self::Fov),
            8 => Some(Self::SimpleRadialFisheye),
            9 => Some(Self::RadialFisheye),
            10 => Some(Self::ThinPrismFisheye),
            _ => None,
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "SIMPLE_PINHOLE" => Some(Self::SimplePinhole),
            "PINHOLE" => Some(Self::Pinhole),
            "SIMPLE_RADIAL" => Some(Self::SimpleRadial),
            "RADIAL" => Some(Self::Radial),
            "OPENCV" => Some(Self::OpenCV),
            "OPENCV_FISHEYE" => Some(Self::OpenCvFishEye),
            "FULL_OPENCV" => Some(Self::FullOpenCV),
            "FOV" => Some(Self::Fov),
            "SIMPLE_RADIAL_FISHEYE" => Some(Self::SimpleRadialFisheye),
            "RADIAL_FISHEYE" => Some(Self::RadialFisheye),
            "THIN_PRISM_FISHEYE" => Some(Self::ThinPrismFisheye),
            _ => None,
        }
    }

    fn num_params(&self) -> usize {
        match self {
            Self::SimplePinhole => 3,
            Self::Pinhole => 4,
            Self::SimpleRadial => 4,
            Self::Radial => 5,
            Self::OpenCV => 8,
            Self::OpenCvFishEye => 8,
            Self::FullOpenCV => 12,
            Self::Fov => 5,
            Self::SimpleRadialFisheye => 4,
            Self::RadialFisheye => 5,
            Self::ThinPrismFisheye => 12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub id: i32,
    pub model: CameraModel,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

#[derive(Debug)]
pub struct Image {
    pub id: i32,
    pub tvec: glam::Vec3,
    pub quat: glam::Quat,
    pub camera_id: i32,
    pub name: String,

    pub points: Option<ImagePointData>,
}

#[derive(Debug)]
pub struct ImagePointData {
    pub xys: Vec<glam::Vec2>,
    pub point3d_ids: Vec<i64>,
}

#[derive(Debug)]
pub struct Point3D {
    pub id: i64,
    pub xyz: glam::Vec3,
    pub rgb: [u8; 3],
    pub aux: Option<Point3DAux>,
}

#[derive(Debug)]
pub struct Point3DAux {
    pub error: f64,
    pub image_ids: Vec<i32>,
    pub point2d_idxs: Vec<i32>,
}

impl Camera {
    pub fn focal(&self) -> (f64, f64) {
        let x = self.params[0];
        let y = self.params[match self.model {
            CameraModel::SimplePinhole => 0,
            CameraModel::Pinhole => 1,
            CameraModel::SimpleRadial => 0,
            CameraModel::Radial => 0,
            CameraModel::OpenCV => 1,
            CameraModel::OpenCvFishEye => 1,
            CameraModel::FullOpenCV => 1,
            CameraModel::Fov => 1,
            CameraModel::SimpleRadialFisheye => 0,
            CameraModel::RadialFisheye => 0,
            CameraModel::ThinPrismFisheye => 1,
        }];
        (x, y)
    }

    pub fn principal_point(&self) -> glam::Vec2 {
        let x = self.params[match self.model {
            CameraModel::SimplePinhole => 1,
            CameraModel::Pinhole => 2,
            CameraModel::SimpleRadial => 1,
            CameraModel::Radial => 1,
            CameraModel::OpenCV => 2,
            CameraModel::OpenCvFishEye => 2,
            CameraModel::FullOpenCV => 2,
            CameraModel::Fov => 2,
            CameraModel::SimpleRadialFisheye => 1,
            CameraModel::RadialFisheye => 1,
            CameraModel::ThinPrismFisheye => 2,
        }] as f32;
        let y = self.params[match self.model {
            CameraModel::SimplePinhole => 2,
            CameraModel::Pinhole => 3,
            CameraModel::SimpleRadial => 2,
            CameraModel::Radial => 2,
            CameraModel::OpenCV => 3,
            CameraModel::OpenCvFishEye => 3,
            CameraModel::FullOpenCV => 3,
            CameraModel::Fov => 3,
            CameraModel::SimpleRadialFisheye => 2,
            CameraModel::RadialFisheye => 2,
            CameraModel::ThinPrismFisheye => 3,
        }] as f32;
        glam::vec2(x, y)
    }
}

fn parse<T: std::str::FromStr>(s: &str) -> io::Result<T> {
    s.parse()
        .map_err(|_e| io::Error::new(io::ErrorKind::InvalidData, "Parse error"))
}

async fn read_cameras_text<R: AsyncBufRead + Unpin>(mut reader: R) -> io::Result<Vec<Camera>> {
    let mut cameras = Vec::new();
    let mut line = String::new();

    while reader.read_line(&mut line).await? > 0 {
        if line.starts_with('#') {
            line.clear();
            continue;
        }

        let parts: Vec<&str> = line.split_ascii_whitespace().collect();
        if parts.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid camera data",
            ));
        }

        let id = parse(parts[0])?;
        let model = CameraModel::from_name(parts[1])
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid camera model"))?;

        let width = parse(parts[2])?;
        let height = parse(parts[3])?;
        let params: Vec<f64> = parts[4..]
            .iter()
            .map(|&s| parse(s))
            .collect::<Result<_, _>>()?;

        if params.len() != model.num_params() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid number of camera parameters (was given {}, but expected {})",
                    params.len(),
                    model.num_params()
                ),
            ));
        }

        cameras.push(Camera {
            id,
            model,
            width,
            height,
            params,
        });
        line.clear();

        tokio_wasm::task::yield_now().await;
    }

    Ok(cameras)
}

async fn read_cameras_binary<R: AsyncRead + Unpin>(mut reader: R) -> io::Result<Vec<Camera>> {
    let mut cameras = Vec::new();
    let num_cameras = reader.read_u64_le().await?;

    for _ in 0..num_cameras {
        let camera_id = reader.read_i32_le().await?;
        let model_id = reader.read_i32_le().await?;
        let width = reader.read_u64_le().await?;
        let height = reader.read_u64_le().await?;

        let model = CameraModel::from_id(model_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid camera model"))?;

        let num_params = model.num_params();
        let mut params = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            params.push(reader.read_f64_le().await?);
        }

        cameras.push(Camera {
            id: camera_id,
            model,
            width,
            height,
            params,
        });
    }

    Ok(cameras)
}

async fn read_images_text<R: AsyncBufRead + Unpin>(
    reader: R,
    with_points: bool,
) -> io::Result<Vec<Image>> {
    let mut images = vec![];
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut elems = line.split_ascii_whitespace();
        let mut try_next = || {
            elems.next().ok_or(io::Error::new(
                io::ErrorKind::InvalidData,
                "Missing element",
            ))
        };

        let id: i32 = parse(try_next()?)?;

        let [w, x, y, z] = [
            parse(try_next()?)?,
            parse(try_next()?)?,
            parse(try_next()?)?,
            parse(try_next()?)?,
        ];
        let quat = glam::quat(x, y, z, w);
        let tvec = glam::vec3(
            parse(try_next()?)?,
            parse(try_next()?)?,
            parse(try_next()?)?,
        );
        let camera_id: i32 = parse(try_next()?)?;
        let name = try_next()?.to_owned();

        let next_line = lines
            .next_line()
            .await?
            .ok_or(io::Error::new(io::ErrorKind::InvalidData, "Missing line"))?;

        let point_data = if with_points {
            let elems: Vec<&str> = next_line.split_whitespace().collect();
            let mut xys = Vec::new();
            let mut point3d_ids = Vec::new();

            for chunk in elems.chunks(3) {
                xys.push(glam::vec2(parse(chunk[0])?, parse(chunk[1])?));
                point3d_ids.push(parse(chunk[2])?);
            }

            Some(ImagePointData { xys, point3d_ids })
        } else {
            None
        };

        images.push(Image {
            id,
            quat,
            tvec,
            camera_id,
            name,
            points: point_data,
        });

        if images.len() % 1000 == 0 {
            log::info!("Read {} image infos", images.len());
        }
    }

    Ok(images)
}

async fn read_images_binary<R: AsyncBufRead + Unpin>(
    mut reader: R,
    with_points: bool,
) -> io::Result<Vec<Image>> {
    let mut images = Vec::new();
    let num_images = reader.read_u64_le().await?;

    for _ in 0..num_images {
        let image_id = reader.read_i32_le().await?;

        let [w, x, y, z] = [
            reader.read_f64_le().await? as f32,
            reader.read_f64_le().await? as f32,
            reader.read_f64_le().await? as f32,
            reader.read_f64_le().await? as f32,
        ];
        let quat = glam::quat(x, y, z, w);

        let tvec = glam::vec3(
            reader.read_f64_le().await? as f32,
            reader.read_f64_le().await? as f32,
            reader.read_f64_le().await? as f32,
        );
        let camera_id = reader.read_i32_le().await?;
        let mut name_bytes = Vec::new();
        reader.read_until(b'\0', &mut name_bytes).await?;

        let name = std::str::from_utf8(&name_bytes[..name_bytes.len() - 1])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .to_owned();

        let num_points2d = reader.read_u64_le().await?;

        let point_data = if with_points {
            let mut xys = Vec::with_capacity(num_points2d as usize);
            let mut point3d_ids = Vec::with_capacity(num_points2d as usize);

            for _ in 0..num_points2d {
                xys.push(glam::Vec2::new(
                    reader.read_f64_le().await? as f32,
                    reader.read_f64_le().await? as f32,
                ));
                point3d_ids.push(reader.read_i64().await?);
            }
            Some(ImagePointData { xys, point3d_ids })
        } else {
            // Advance reader correct amount.
            for _ in 0..num_points2d {
                let (_, _, _) = (
                    reader.read_f64_le().await?,
                    reader.read_f64_le().await?,
                    reader.read_i64().await?,
                );
            }
            None
        };

        images.push(Image {
            id: image_id,
            quat,
            tvec,
            camera_id,
            name,
            points: point_data,
        });
    }

    Ok(images)
}

async fn read_points3d_text<R: AsyncBufRead + Unpin>(
    reader: R,
    with_aux: bool,
) -> io::Result<Vec<Point3D>> {
    let mut points3d = Vec::new();
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        if line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_ascii_whitespace();

        let mut try_next = || {
            parts.next().ok_or(io::Error::new(
                io::ErrorKind::InvalidData,
                "Missing element",
            ))
        };

        let id: i64 = parse(try_next()?)?;
        let xyz = glam::Vec3::new(
            parse::<f32>(try_next()?)?,
            parse::<f32>(try_next()?)?,
            parse::<f32>(try_next()?)?,
        );
        let rgb = [
            parse::<u8>(try_next()?)?,
            parse::<u8>(try_next()?)?,
            parse::<u8>(try_next()?)?,
        ];

        let points_aux = if with_aux {
            let error: f64 = parse(try_next()?)?;

            let mut image_ids = Vec::new();
            let mut point2d_idxs = Vec::new();

            loop {
                let (id, idx_2d) = (try_next(), try_next());
                match (id, idx_2d) {
                    (Ok(id), Ok(idx_2d)) => {
                        image_ids.push(parse(id)?);
                        point2d_idxs.push(parse(idx_2d)?);
                    }
                    (Ok(_), Err(b)) => {
                        Err(b)?;
                    }
                    _ => break,
                }
            }

            Some(Point3DAux {
                error,
                image_ids,
                point2d_idxs,
            })
        } else {
            None
        };

        if id % 100000 == 0 {
            log::info!("Processed {id} points");
        }

        points3d.push(Point3D {
            id,
            xyz,
            rgb,
            aux: points_aux,
        });
    }

    Ok(points3d)
}

async fn read_points3d_binary<R: AsyncRead + Unpin>(
    mut reader: R,
    points_aux: bool,
) -> io::Result<Vec<Point3D>> {
    let mut points3d = Vec::new();
    let num_points = reader.read_u64_le().await?;

    for _ in 0..num_points {
        let point3d_id = reader.read_i64().await?;
        let xyz = glam::Vec3::new(
            reader.read_f64_le().await? as f32,
            reader.read_f64_le().await? as f32,
            reader.read_f64_le().await? as f32,
        );
        let rgb = [
            reader.read_u8().await?,
            reader.read_u8().await?,
            reader.read_u8().await?,
        ];

        let error = reader.read_f64_le().await?;
        let track_length = reader.read_u64_le().await?;

        let points_aux = if points_aux {
            let mut image_ids = Vec::with_capacity(track_length as usize);
            let mut point2d_idxs = Vec::with_capacity(track_length as usize);

            for _ in 0..track_length {
                image_ids.push(reader.read_i32_le().await?);
                point2d_idxs.push(reader.read_i32_le().await?);
            }

            Some(Point3DAux {
                error,
                image_ids,
                point2d_idxs,
            })
        } else {
            for _ in 0..track_length {
                let _ = reader.read_i32_le().await?;
                let _ = reader.read_i32_le().await?;
            }
            None
        };

        points3d.push(Point3D {
            id: point3d_id,
            xyz,
            rgb,
            aux: points_aux,
        });
    }

    Ok(points3d)
}

pub async fn read_cameras<R: AsyncBufRead + Unpin>(
    reader: R,
    binary: bool,
) -> io::Result<Vec<Camera>> {
    if binary {
        read_cameras_binary(reader).await
    } else {
        read_cameras_text(reader).await
    }
}

pub async fn read_images<R: AsyncBufRead + Unpin>(
    reader: R,
    binary: bool,
    with_points: bool,
) -> io::Result<Vec<Image>> {
    if binary {
        read_images_binary(reader, with_points).await
    } else {
        read_images_text(reader, with_points).await
    }
}

pub async fn read_points3d<R: AsyncBufRead + Unpin>(
    reader: R,
    binary: bool,
    points_aux: bool,
) -> io::Result<Vec<Point3D>> {
    if binary {
        read_points3d_binary(reader, points_aux).await
    } else {
        read_points3d_text(reader, points_aux).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tokio::io::BufReader;

    #[test]
    fn test_camera_model_workflow() {
        // Test camera model parsing and parameter extraction
        let models = [
            (0, "SIMPLE_PINHOLE", 3),
            (1, "PINHOLE", 4),
            (4, "OPENCV", 8),
            (6, "FULL_OPENCV", 12),
        ];

        for (id, name, expected_params) in models {
            let from_id = CameraModel::from_id(id).unwrap();
            let from_name = CameraModel::from_name(name).unwrap();
            assert_eq!(from_id.num_params(), expected_params);
            assert_eq!(from_name.num_params(), expected_params);
        }

        // Test invalid cases
        assert!(CameraModel::from_id(99).is_none());
        assert!(CameraModel::from_name("INVALID").is_none());
    }

    #[test]
    fn test_camera_intrinsics() {
        let pinhole_camera = Camera {
            id: 1,
            model: CameraModel::Pinhole,
            width: 640,
            height: 480,
            params: vec![500.0, 501.0, 320.0, 240.0],
        };

        assert_eq!(pinhole_camera.focal(), (500.0, 501.0));
        let pp = pinhole_camera.principal_point();
        assert_eq!(pp.x, 320.0);
        assert_eq!(pp.y, 240.0);

        let simple_camera = Camera {
            id: 2,
            model: CameraModel::SimplePinhole,
            width: 640,
            height: 480,
            params: vec![500.0, 320.0, 240.0],
        };
        assert_eq!(simple_camera.focal(), (500.0, 500.0));
    }

    #[tokio::test]
    async fn test_camera_parsing_workflow() {
        let camera_data = "# Camera list with one line of data per camera:\n\
                          # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\
                          1 PINHOLE 800 600 500.0 500.0 400.0 300.0\n\
                          # Comments should be ignored\n\
                          2 OPENCV 640 480 450.0 451.0 320.0 240.0 0.1 0.2 0.3 0.4\n";

        let reader = Cursor::new(camera_data.as_bytes());
        let cameras = read_cameras_text(reader).await.unwrap();

        assert_eq!(cameras.len(), 2);
        let cam1 = &cameras[0];
        assert_eq!(cam1.params, vec![500.0, 500.0, 400.0, 300.0]);
        assert_eq!(cam1.focal(), (500.0, 500.0));

        let cam2 = &cameras[1];
        assert_eq!(cam2.params.len(), 8);
        assert!(matches!(cam2.model, CameraModel::OpenCV));

        // Test error cases - should fail
        let invalid_model = "1 INVALID_MODEL 800 600 500.0 500.0 400.0 300.0\n";
        let reader = Cursor::new(invalid_model.as_bytes());
        let result = read_cameras_text(reader).await;
        assert!(result.is_err());

        let wrong_params = "1 PINHOLE 800 600 500.0 500.0 400.0\n"; // Missing one param
        let reader = Cursor::new(wrong_params.as_bytes());
        let result = read_cameras_text(reader).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_images_parsing_workflow() {
        let image_data = "# Image list with two lines of data per image:\n\
                         1 0.7071 0.0 0.0 0.7071 1.0 2.0 3.0 1 image1.jpg\n\
                         100.0 200.0 1 150.0 250.0 2 200.0 300.0 -1\n\
                         2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 image2.jpg\n\
                         \n";

        let reader = Cursor::new(image_data.as_bytes());
        let images = read_images_text(reader, true).await.unwrap();

        assert_eq!(images.len(), 2);
        let img1 = &images[0];
        assert_eq!(img1.camera_id, 1);
        assert_eq!(img1.name, "image1.jpg");
        assert_eq!(img1.points.as_ref().unwrap().xys.len(), 3);
        assert_eq!(img1.points.as_ref().unwrap().point3d_ids[2], -1); // Invalid point3d id
        let img2 = &images[1];
        assert_eq!(img2.points.as_ref().unwrap().xys.len(), 0); // No 2D points
    }

    #[tokio::test]
    async fn test_points3d_parsing_workflow() {
        let points_data = "# 3D point list\n\
                          1 1.5 2.5 3.5 255 128 64 0.1 1 100 2 200\n\
                          2 -1.0 0.0 1.0 0 255 0 0.05 3 50 4 75 5 125\n";

        let reader = Cursor::new(points_data.as_bytes());
        let points = read_points3d_text(reader, true).await.unwrap();

        assert_eq!(points.len(), 2);
        let pt1 = &points[0];
        assert_eq!(pt1.xyz, glam::vec3(1.5, 2.5, 3.5));
        assert_eq!(pt1.rgb, [255, 128, 64]);
        assert_eq!(pt1.aux.as_ref().unwrap().image_ids, vec![1, 2]);

        // Test error case - should fail
        let invalid_data = "1 1.5 2.5 3.5 255 128 64 0.1 1\n"; // Missing POINT2D_IDX
        let reader = Cursor::new(invalid_data.as_bytes());
        let result = read_points3d_text(reader, true).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_error_handling_workflow() {
        // Test various malformed inputs - these should all fail
        let malformed_cases = [
            ("1 PINHOLE 800\n", "cameras"), // Too few fields
            ("1 PINHOLE abc 600 500.0 500.0 400.0 300.0\n", "cameras"), // Non-numeric
            ("1 1.5 2.5\n", "points3d"),    // Too few fields
        ];

        for (data, data_type) in malformed_cases {
            let reader = Cursor::new(data.as_bytes());
            let result = match data_type {
                "cameras" => read_cameras_text(reader).await.map(|_| ()),
                "points3d" => read_points3d_text(reader, false).await.map(|_| ()),
                _ => unreachable!(),
            };
            assert!(result.is_err(), "Expected error for: {data}");
        }

        // Test empty files work
        let reader = Cursor::new(b"");
        let cameras = read_cameras_text(reader).await.unwrap();
        assert_eq!(cameras.len(), 0);
    }

    #[tokio::test]
    async fn test_public_api_integration() {
        let camera_data = "1 PINHOLE 800 600 500.0 500.0 400.0 300.0\n";
        let reader = Cursor::new(camera_data.as_bytes());
        let cameras = read_cameras(reader, false).await.unwrap();
        assert_eq!(cameras.len(), 1);
        let image_data = "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 test.jpg\n\n";
        let reader = BufReader::new(Cursor::new(image_data.as_bytes()));
        let images = read_images(reader, false, false).await.unwrap();
        assert_eq!(images.len(), 1);
        let points_data = "1 1.0 2.0 3.0 255 0 0 0.1\n";
        let reader = Cursor::new(points_data.as_bytes());
        let points = read_points3d(reader, false, true).await.unwrap();
        assert_eq!(points.len(), 1);
        // Verify data consistency
        let camera = &cameras[0];
        let image = &images[0];
        assert_eq!(image.camera_id, camera.id);
    }
}
