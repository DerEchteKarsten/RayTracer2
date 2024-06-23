use bevy::{input::{mouse::{MouseButtonInput, MouseMotion}, ButtonState}, prelude::*, window::{CursorGrabMode, PrimaryWindow}};

use glam::{vec3, Mat3, Mat4, Quat, Vec3, Vec4};

const MOVE_SPEED: f32 = 2.0;
const ANGLE_PER_POINT: f32 = 0.0009;

const UP: Vec3 = vec3(0.0, 1.0, 0.0);

#[derive(Debug, Default, Clone, Copy, PartialEq, Resource)]
pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub fov: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
}

#[derive(Clone, Copy, Resource, Default)]
pub struct CameraUniformData {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub input: Vec4,
}

impl Camera {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        fov: f32,
        aspect_ratio: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            fov,
            aspect_ratio,
            z_near,
            z_far,
        }
    }
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(
            self.position,
            self.position + self.direction,
            vec3(0.0, 1.0, 0.0),
        )
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

pub fn update_camera(mut camera: ResMut<Camera>, controls: Res<Controls>, time: Res<Time>) {
    let delta_time = time.delta_seconds();
    let side = camera.direction.cross(UP);

    // Update direction
    let new_direction = if controls.look_around {
        let side_rot = Quat::from_axis_angle(side, -controls.cursor_delta[1] * ANGLE_PER_POINT);
        let y_rot = Quat::from_rotation_y(-controls.cursor_delta[0] * ANGLE_PER_POINT);
        let rot = Mat3::from_quat(side_rot * y_rot);

        (rot * camera.direction).normalize()
    } else {
        camera.direction
    };

    // Update position
    let mut direction = Vec3::ZERO;

    if controls.go_forward {
        direction += new_direction;
    }
    if controls.go_backward {
        direction -= new_direction;
    }
    if controls.strafe_right {
        direction += side;
    }
    if controls.strafe_left {
        direction -= side;
    }
    if controls.go_up {
        direction += UP;
    }
    if controls.go_down {
        direction -= UP;
    }

    let direction = if direction.length_squared() == 0.0 {
        direction
    } else {
        direction.normalize()
    };

    camera.position += direction * MOVE_SPEED * delta_time;
    camera.direction = new_direction;
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
    for ev in mousebtn_evr.read() {
        if ev.button == MouseButton::Right && ev.state == ButtonState::Pressed {
            controls.look_around = true;
            windows.single_mut().cursor.grab_mode = CursorGrabMode::None;
        }
    }
}
pub fn update_mouse_move(
    mut controls: ResMut<Controls>,
    mut evr_motion: EventReader<MouseMotion>,
) {
    for ev in evr_motion.read() {
        controls.cursor_delta = [
            controls.cursor_delta[0] + ev.delta.x,
            controls.cursor_delta[1] + ev.delta.y,
        ];
    }
}

pub fn reset(mut controls: ResMut<Controls>) {
    controls.cursor_delta = [0.0, 0.0];
}

pub fn update_keyboard(keys: Res<ButtonInput<KeyCode>>, mut controls: ResMut<Controls>,) {
    controls.go_forward = keys.pressed(KeyCode::KeyW);
    controls.go_backward = keys.pressed(KeyCode::KeyS);
    controls.strafe_right = keys.pressed(KeyCode::KeyD);
    controls.strafe_left = keys.pressed(KeyCode::KeyA);
    controls.go_up = keys.pressed(KeyCode::Space);
    controls.go_down = keys.pressed(KeyCode::ShiftLeft);
}

pub fn CameraPlugin(app: &mut App) {
    app.init_resource::<Camera>()
        .init_resource::<Controls>()
        .add_systems(PreUpdate, reset)
        .add_systems(Update, (update_camera, update_mouse_move, update_mouse_buttons, update_keyboard));
}
