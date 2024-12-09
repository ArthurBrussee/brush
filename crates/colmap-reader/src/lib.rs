#![allow(unused)]

use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, BufRead, Read};
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncReadExt;
use tokio::io::{AsyncBufRead, AsyncRead};

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
            0 => Some(CameraModel::SimplePinhole),
            1 => Some(CameraModel::Pinhole),
            2 => Some(CameraModel::SimpleRadial),
            3 => Some(CameraModel::Radial),
            4 => Some(CameraModel::OpenCV),
            5 => Some(CameraModel::OpenCvFishEye),
            6 => Some(CameraModel::FullOpenCV),
            7 => Some(CameraModel::Fov),
            8 => Some(CameraModel::SimpleRadialFisheye),
            9 => Some(CameraModel::RadialFisheye),
            10 => Some(CameraModel::ThinPrismFisheye),
            _ => None,
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "SIMPLE_PINHOLE" => Some(CameraModel::SimplePinhole),
            "PINHOLE" => Some(CameraModel::Pinhole),
            "SIMPLE_RADIAL" => Some(CameraModel::SimpleRadial),
            "RADIAL" => Some(CameraModel::Radial),
            "OPENCV" => Some(CameraModel::OpenCV),
            "OPENCV_FISHEYE" => Some(CameraModel::OpenCvFishEye),
            "FULL_OPENCV" => Some(CameraModel::FullOpenCV),
            "FOV" => Some(CameraModel::Fov),
            "SIMPLE_RADIAL_FISHEYE" => Some(CameraModel::SimpleRadialFisheye),
            "RADIAL_FISHEYE" => Some(CameraModel::RadialFisheye),
            "THIN_PRISM_FISHEYE" => Some(CameraModel::ThinPrismFisheye),
            _ => None,
        }
    }

    fn num_params(&self) -> usize {
        match self {
            CameraModel::SimplePinhole => 3,
            CameraModel::Pinhole => 4,
            CameraModel::SimpleRadial => 4,
            CameraModel::Radial => 5,
            CameraModel::OpenCV => 8,
            CameraModel::OpenCvFishEye => 8,
            CameraModel::FullOpenCV => 12,
            CameraModel::Fov => 5,
            CameraModel::SimpleRadialFisheye => 4,
            CameraModel::RadialFisheye => 5,
            CameraModel::ThinPrismFisheye => 12,
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
    pub tvec: glam::Vec3,
    pub quat: glam::Quat,
    pub camera_id: i32,
    pub name: String,
    pub xys: Vec<glam::Vec2>,
    pub point3d_ids: Vec<i64>,
}

#[derive(Debug)]
pub struct Point3D {
    pub xyz: glam::Vec3,
    pub rgb: [u8; 3],
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
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Parse error"))
}

async fn read_cameras_text<R: AsyncRead + Unpin>(reader: R) -> io::Result<HashMap<i32, Camera>> {
    let mut cameras = HashMap::new();
    let mut buf_reader = tokio::io::BufReader::new(reader);
    let mut line = String::new();

    while buf_reader.read_line(&mut line).await? > 0 {
        if line.starts_with('#') {
            line.clear();
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
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
                "Invalid number of camera parameters",
            ));
        }

        cameras.insert(
            id,
            Camera {
                id,
                model,
                width,
                height,
                params,
            },
        );
        line.clear();
    }

    Ok(cameras)
}

async fn read_cameras_binary<R: AsyncRead + Unpin>(
    mut reader: R,
) -> io::Result<HashMap<i32, Camera>> {
    let mut cameras = HashMap::new();
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

        cameras.insert(
            camera_id,
            Camera {
                id: camera_id,
                model,
                width,
                height,
                params,
            },
        );
    }

    Ok(cameras)
}

async fn read_images_text<R: AsyncRead + Unpin>(mut reader: R) -> io::Result<HashMap<i32, Image>> {
    let mut images = HashMap::new();
    let mut buf_reader = tokio::io::BufReader::new(reader);
    let mut line = String::new();

    let mut img_data = true;

    loop {
        line.clear();
        if buf_reader.read_line(&mut line).await? == 0 {
            break;
        }

        if !line.is_empty() && !line.starts_with('#') {
            let elems: Vec<&str> = line.split_whitespace().collect();
            let id: i32 = parse(elems[0])?;

            let [w, x, y, z] = [
                parse(elems[1])?,
                parse(elems[2])?,
                parse(elems[3])?,
                parse(elems[4])?,
            ];
            let quat = glam::quat(x, y, z, w);
            let tvec = glam::vec3(parse(elems[5])?, parse(elems[6])?, parse(elems[7])?);
            let camera_id: i32 = parse(elems[8])?;
            let name = elems[9].to_string();

            line.clear();
            buf_reader.read_line(&mut line).await?;
            let elems: Vec<&str> = line.split_whitespace().collect();
            let mut xys = Vec::new();
            let mut point3d_ids = Vec::new();

            for chunk in elems.chunks(3) {
                xys.push(glam::vec2(parse(chunk[0])?, parse(chunk[1])?));
                point3d_ids.push(parse(chunk[2])?);
            }

            images.insert(
                id,
                Image {
                    quat,
                    tvec,
                    camera_id,
                    name,
                    xys,
                    point3d_ids,
                },
            );
        }
    }

    Ok(images)
}

async fn read_images_binary<R: AsyncBufRead + Unpin>(
    mut reader: R,
) -> io::Result<HashMap<i32, Image>> {
    let mut images = HashMap::new();
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
        let mut xys = Vec::with_capacity(num_points2d as usize);
        let mut point3d_ids = Vec::with_capacity(num_points2d as usize);

        for _ in 0..num_points2d {
            xys.push(glam::Vec2::new(
                reader.read_f64_le().await? as f32,
                reader.read_f64_le().await? as f32,
            ));
            point3d_ids.push(reader.read_i64().await?);
        }

        images.insert(
            image_id,
            Image {
                quat,
                tvec,
                camera_id,
                name,
                xys,
                point3d_ids,
            },
        );
    }

    Ok(images)
}

async fn read_points3d_text<R: AsyncRead + Unpin>(
    mut reader: R,
) -> io::Result<HashMap<i64, Point3D>> {
    let mut points3d = HashMap::new();
    let mut buf_reader = tokio::io::BufReader::new(reader);
    let mut line = String::new();

    while buf_reader.read_line(&mut line).await? > 0 {
        if line.starts_with('#') {
            line.clear();
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid point3D data",
            ));
        }

        let id: i64 = parse(parts[0])?;
        let xyz = glam::Vec3::new(parse(parts[1])?, parse(parts[2])?, parse(parts[3])?);
        let rgb = [
            parse::<u8>(parts[4])?,
            parse::<u8>(parts[5])?,
            parse::<u8>(parts[6])?,
        ];
        let error: f64 = parse(parts[7])?;

        let mut image_ids = Vec::new();
        let mut point2d_idxs = Vec::new();

        for chunk in parts[8..].chunks(2) {
            if chunk.len() < 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid point3D track data",
                ));
            }
            image_ids.push(parse(chunk[0])?);
            point2d_idxs.push(parse(chunk[1])?);
        }

        points3d.insert(
            id,
            Point3D {
                xyz,
                rgb,
                error,
                image_ids,
                point2d_idxs,
            },
        );
        line.clear();
    }

    Ok(points3d)
}

async fn read_points3d_binary<R: AsyncRead + Unpin>(
    mut reader: R,
) -> io::Result<HashMap<i64, Point3D>> {
    let mut points3d = HashMap::new();
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
        let mut image_ids = Vec::with_capacity(track_length as usize);
        let mut point2d_idxs = Vec::with_capacity(track_length as usize);

        for _ in 0..track_length {
            image_ids.push(reader.read_i32_le().await?);
            point2d_idxs.push(reader.read_i32_le().await?);
        }

        points3d.insert(
            point3d_id,
            Point3D {
                xyz,
                rgb,
                error,
                image_ids,
                point2d_idxs,
            },
        );
    }

    Ok(points3d)
}

pub async fn read_cameras<R: AsyncRead + Unpin>(
    mut reader: R,
    binary: bool,
) -> io::Result<HashMap<i32, Camera>> {
    if binary {
        read_cameras_binary(reader).await
    } else {
        read_cameras_text(reader).await
    }
}

pub async fn read_images<R: AsyncBufRead + Unpin>(
    reader: R,
    binary: bool,
) -> io::Result<HashMap<i32, Image>> {
    if binary {
        read_images_binary(reader).await
    } else {
        read_images_text(reader).await
    }
}

pub async fn read_points3d<R: AsyncRead + Unpin>(
    reader: R,
    binary: bool,
) -> io::Result<HashMap<i64, Point3D>> {
    if binary {
        read_points3d_binary(reader).await
    } else {
        read_points3d_text(reader).await
    }
}
