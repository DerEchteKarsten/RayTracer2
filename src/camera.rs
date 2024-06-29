use bevy::{
    input::{
        mouse::{MouseButtonInput, MouseMotion},
        ButtonState,
    },
    prelude::*,
    time::Time,
    window::{CursorGrabMode, PrimaryWindow},
};

use glam::{vec3, Mat3, Mat4, Quat, Vec3, Vec3Swizzles, Vec4};

use crate::{
    components::{PhysicsBody, Player, Position},
    WINDOW_SIZE,
};
#[derive(Debug, Default, Clone, Copy, PartialEq, Component)]
pub struct Camera {
    pub fov: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub is_main: bool,
}

#[derive(Clone, Copy, Resource, Default)]
pub struct CameraUniformData {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub input: Vec4,
}

impl Camera {
    pub fn new(fov: f32, aspect_ratio: f32, z_near: f32, z_far: f32, is_main: bool) -> Self {
        Self {
            fov,
            aspect_ratio,
            z_near,
            z_far,
            is_main,
        }
    }
    pub fn view_matrix(position: Vec3, direction: Vec3) -> Mat4 {
        Mat4::look_at_rh(position, position + direction, vec3(0.0, 1.0, 0.0))
    }

    pub fn projection_matrix(&self) -> Mat4 {
        perspective(
            self.fov.to_radians(),
            self.aspect_ratio,
            self.z_near,
            self.z_far,
        )
    }
}

#[rustfmt::skip]
pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    
    let f = (fovy / 2.0).tan().recip();

    let c0r0 = f / aspect;
    let c0r1 = 0.0f32;
    let c0r2 = 0.0f32;
    let c0r3 = 0.0f32;

    let c1r0 = 0.0f32;
    let c1r1 = -f;
    let c1r2 = 0.0f32;
    let c1r3 = 0.0f32;

    let c2r0 = 0.0f32;
    let c2r1 = 0.0f32;
    let c2r2 = -far / (far - near);
    let c2r3 = -1.0f32;

    let c3r0 = 0.0f32;
    let c3r1 = 0.0f32;
    let c3r2 = -(far * near) / (far - near);
    let c3r3 = 0.0f32;

    Mat4::from_cols_array(&[
        c0r0, c0r1, c0r2, c0r3,
        c1r0, c1r1, c1r2, c1r3,
        c2r0, c2r1, c2r2, c2r3,
        c3r0, c3r1, c3r2, c3r3
    ])
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct Controls {
    pub go_forward: bool,
    pub go_backward: bool,
    pub strafe_right: bool,
    pub strafe_left: bool,
    pub go_up: bool,
    pub go_down: bool,
    pub look_around: bool,
    pub cursor_delta: [f32; 2],
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            go_forward: false,
            go_backward: false,
            strafe_right: false,
            strafe_left: false,
            go_up: false,
            go_down: false,
            look_around: false,
            cursor_delta: [0.0; 2],
        }
    }
}

pub fn update_mouse_buttons(
    mut controls: ResMut<Controls>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut mousebtn_evr: EventReader<MouseButtonInput>,
) {
    let mut window = windows.single_mut();
    for ev in mousebtn_evr.read() {
        if ev.button == MouseButton::Right && ev.state == ButtonState::Pressed {
            controls.look_around = true;
            window.cursor.grab_mode = CursorGrabMode::Confined;
            window.cursor.visible = false;
        }
        if ev.button == MouseButton::Right && ev.state == ButtonState::Released {
            controls.look_around = false;
            window.cursor.grab_mode = CursorGrabMode::None;
            window.cursor.visible = true;
        }
    }
}
pub fn update_mouse_move(mut controls: ResMut<Controls>, mut evr_motion: EventReader<MouseMotion>) {
    for ev in evr_motion.read() {
        controls.cursor_delta = [
            controls.cursor_delta[0] + ev.delta.x,
            controls.cursor_delta[1] + ev.delta.y,
        ];
    }
}

pub fn update_keyboard(keys: Res<ButtonInput<KeyCode>>, mut controls: ResMut<Controls>) {
    controls.go_forward = keys.pressed(KeyCode::KeyW);
    controls.go_backward = keys.pressed(KeyCode::KeyS);
    controls.strafe_right = keys.pressed(KeyCode::KeyD);
    controls.strafe_left = keys.pressed(KeyCode::KeyA);
    controls.go_up = keys.pressed(KeyCode::Space);
    controls.go_down = keys.pressed(KeyCode::ShiftLeft);
}

fn update_camera_matrix(
    query: Query<(&Camera, &Position)>,
    mut uniform_data: ResMut<CameraUniformData>,
    mut controls: ResMut<Controls>,
) {
    for (camera, position) in &query {
        if camera.is_main {
            uniform_data.proj_inverse = camera.projection_matrix().inverse();
            uniform_data.view_inverse =
                Camera::view_matrix(position.position, position.rotation).inverse();
            uniform_data.input.z = WINDOW_SIZE.0 as f32;
            uniform_data.input.w = WINDOW_SIZE.1 as f32;
        }
    }
    controls.cursor_delta = [0.0, 0.0];
}

pub fn CameraPlugin(app: &mut App) {
    app.init_resource::<Controls>().add_systems(
        PreUpdate,
        (
            update_camera_matrix.before(update_mouse_move),
            update_mouse_move,
            update_mouse_buttons,
            update_keyboard,
        ),
    );
}
