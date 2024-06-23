use bevy::{app::DynEq, prelude::*};
use glam::*;

use crate::{components::{PhysicsBody, Player, Position}, oct_tree::{ray_voxel, GameWorld, Octant}, Controls};
const MOVE_SPEED: f32 = 4.0;
const ANGLE_PER_POINT: f32 = 0.0009;
const UP: glam::Vec3 = vec3(0.0, 1.0, 0.0);
const AIR_SPEED: f32 = 0.02;


pub fn velocity(mut query: Query<(&mut Position, &mut PhysicsBody)>, time: Res<Time>, world: Res<GameWorld>) {
    let delta_time = time.delta_seconds();
    for (mut position, mut vel) in &mut query {
        if vel.velocity.length() != 0.0 {
            let colision = ray_voxel(&world.build_tree, vec3(position.position.x, position.position.y, position.position.z), vel.velocity.normalize());
            vel.grounded = colision.is_some() || position.position.y <= 1.51;
            if let Some(new_pos) = colision {
                position.position = new_pos - vec3(1.5, );
                println!("{}", new_pos);
            }
        }
        if position.position.y + (vel.velocity * delta_time).y < 1.5 {
            position.position.y = 1.5;
            vel.velocity.y = 0.0;
        } else {
            if !vel.grounded {
                vel.velocity.y -= 12.0 * delta_time;
            } 
            position.position += vel.velocity * delta_time;
        }
        // info!("{}", position.position);
    }
}

pub fn movement(mut query: Query<(&mut Position, &mut PhysicsBody), With<Player>>, controls: Res<Controls>) {
    for (mut position, mut pb) in &mut query {
        let side = position.rotation.cross(UP);

        // Update direction
        let new_direction = if controls.look_around {
            let side_rot = glam::Quat::from_axis_angle(side, -controls.cursor_delta[1] * ANGLE_PER_POINT);
            let y_rot = glam::Quat::from_rotation_y(-controls.cursor_delta[0] * ANGLE_PER_POINT);
            let rot = glam::Mat3::from_quat(side_rot * y_rot);

            (rot *position.rotation).normalize()
        } else {
            position.rotation
        };

        // Update position
        let mut direction = glam::Vec3::ZERO;

        if controls.go_forward {
            direction.x += new_direction.x;
            direction.z += new_direction.z;
        }
        if controls.go_backward {
            direction.x -= new_direction.x;
            direction.z -= new_direction.z;
        }
        if controls.strafe_right {
            direction.x += side.x;
            direction.z += side.z;
        }
        if controls.strafe_left {
            direction.x -= side.x;
            direction.z -= side.z;
        }


        let direction = if direction.length_squared() == 0.0 {
            direction
        } else {
            direction.normalize()
        };
        let move_vec = direction * MOVE_SPEED;
        pb.velocity.x = move_vec.x;
        pb.velocity.z = move_vec.z; //if pb.grounded {MOVE_SPEED} else {AIR_SPEED}
        if controls.go_up && pb.grounded {
            pb.velocity.y = 4.0;
        }
        position.rotation = new_direction;
    }
}
pub fn PlayerPlugin(app: &mut App) {
    app.add_systems(Update, movement)
    .add_systems(PostUpdate, velocity);
}