mod render_system;
mod renderer;
use render_system::RenderPlugin;
use renderer::*;
mod camera;
use camera::*;
mod oct_tree;
mod pipelines;

use bevy::a11y::{AccessibilityPlugin, AccessibilityRequested};
use bevy::input::InputPlugin;
use bevy::prelude::*;

use bevy::time::TimePlugin;
use bevy::window::{ExitCondition, WindowResolution};
use bevy::winit::WinitPlugin;

use glam::vec3;
use std::default::Default;

const APP_NAME: &'static str = "Test";
const WINDOW_SIZE: (u32, u32) = (1000, 500);

fn main() {
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
        .insert_resource(Camera::new(
            vec3(1.5, 1.25, -5.0),
            vec3(0.0, 0.0, 1.0),
            40.0,
            WINDOW_SIZE.0 as f32 / WINDOW_SIZE.1 as f32,
            0.1,
            1000.0,
        ))
        .init_resource::<Controls>()
        .init_resource::<CameraUniformData>()
        .add_plugins((
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
        ))
        .run();
}