mod render_system;
mod renderer;
use bevy::log::LogPlugin;
use components::{PhysicsBody, Player, Position};
use oct_tree::{ray_voxel, GameWorld};
use render_system::RenderPlugin;
use renderer::*;
mod camera;
use camera::*;
mod oct_tree;
mod pipelines;
mod components;
mod player;
use player::PlayerPlugin;

use bevy::a11y::{AccessibilityPlugin, AccessibilityRequested};
use bevy::input::InputPlugin;
use bevy::prelude::*;

use bevy::time::TimePlugin;
use bevy::window::{ExitCondition, WindowResolution};
use bevy::winit::WinitPlugin;

use glam::{vec2, vec3, Vec3, Vec4Swizzles};
use std::default::Default;

const APP_NAME: &'static str = "Test";
const WINDOW_SIZE: (u32, u32) = (2000, 1000);

fn setup(mut commands: Commands) {
    let position =  vec3(1.5, 10.0, 1.5);

    commands.spawn((
        Camera::new(
            40.0,
            WINDOW_SIZE.0 as f32 / WINDOW_SIZE.1 as f32,
            0.1,
            1000.0,
            true,
        ), Position {
            position,
            rotation: vec3(0.0, 0.0, 1.0),
        },
        PhysicsBody{
            grounded: false,
            velocity: Vec3::ZERO,
        },
        Player
    ));
}

fn render(cam: Res<CameraUniformData>, world: Res<GameWorld>) {
    let mut imgbuf = image::ImageBuffer::new(WINDOW_SIZE.0, WINDOW_SIZE.1);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let pixelCenter = glam::vec2(x as f32, y as f32) + glam::Vec2::splat(0.5);
        let inUV = pixelCenter/vec2(WINDOW_SIZE.0 as f32, WINDOW_SIZE.1 as f32);
        let d = inUV * 2.0 - 1.0;
        let origin = cam.view_inverse * glam::vec4(0.0,0.0,0.0,1.0);
        let target = (cam.proj_inverse * glam::vec4(d.x, d.y, 1.0, 1.0)).xyz().normalize();
        let direction = cam.view_inverse*glam::vec4(target.x, target.y, target.z, 0.0);
        
        let mut depth = f32::INFINITY;
        world.tree.trace(origin.xyz(), direction.xyz(), world.level_dim, vec3(1.5, 1.5, 1.5), &mut depth);
        
        *pixel = image::Rgb([(f32::min(depth / 10.0, 1.0) * 256.0) as u8, (f32::min(depth / 10.0, 1.0) * 256.0) as u8, (f32::min(depth / 10.0, 1.0) * 256.0) as u8]);
    }

    imgbuf.save("test2.png").unwrap();
    panic!()
}

fn main() {
    let model = oct_tree::Octant::load("./models/map.vox").unwrap();
    App::new()
        .insert_resource(AccessibilityRequested::default())
        .insert_resource(DeviceFeatures {
            ray_tracing_pipeline: false,
            acceleration_structure: false,
            runtime_descriptor_array: true,
            buffer_device_address: true,
            dynamic_rendering: true,
            synchronization2: true,
        })
        .init_resource::<Controls>()
        .init_resource::<CameraUniformData>()
        .insert_resource(GameWorld {tree: model.0.clone(), level_dim: model.1, build_tree: model.0.build()})
        .add_plugins((
            LogPlugin {
                filter: "".to_owned(),
                level: bevy::log::Level::DEBUG,
                ..Default::default()
            },
            AccessibilityPlugin,
            InputPlugin,
            WindowPlugin {
                close_when_requested: true,
                exit_condition: ExitCondition::OnPrimaryClosed,
                primary_window: Some(Window {
                    resolution: WindowResolution::new(WINDOW_SIZE.0 as f32, WINDOW_SIZE.1 as f32),
                    present_mode: bevy::window::PresentMode::AutoNoVsync,
                    title: APP_NAME.to_owned(),
                    ..Default::default()
                }),
            },
            WinitPlugin {
                run_on_any_thread: true,
            },
            CameraPlugin,
            TimePlugin,
            RenderPlugin,
            PlayerPlugin,
        ))
        .add_systems(Startup, setup)
        // .add_systems(PostUpdate, render)
        .run();
}