mod render_system;
mod renderer;
use bevy::log::LogPlugin;
use components::{PhysicsBody, Player, Position};
use oct_tree::GameWorld;
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

use glam::{vec3, Vec3};
use std::default::Default;

const APP_NAME: &'static str = "Test";
const WINDOW_SIZE: (u32, u32) = (2000, 1000);

fn setup(mut commands: Commands) {
    let position =  vec3(1.5, 2.0, -5.0);
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

fn main() {
    let model = oct_tree::Octant::load("./models/monu2.vox").unwrap();
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
                    present_mode: bevy::window::PresentMode::Fifo,
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
        .run();
}