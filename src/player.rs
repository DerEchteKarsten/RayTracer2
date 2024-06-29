use std::f32::consts::PI;

use bevy::{app::DynEq, prelude::*};
use glam::{Vec3Swizzles, Vec4Swizzles, *};

use crate::{
    components::{PhysicsBody, Player, Position},
    oct_tree::{ray_voxel, GameWorld, Octant},
    render_system::GizzmoBuffer,
    Camera, CameraUniformData, Controls,
};
const MOVE_SPEED: f32 = 10.0 * PLAYER_HEIGHT;
const ANGLE_PER_POINT: f32 = 0.0009;
const UP: glam::Vec3 = vec3(0.0, 1.0, 0.0);
const AIR_SPEED: f32 = MOVE_SPEED / 2.0;
const MAX_BOUNCES: u32 = 5;
const SKIN_WIDTH: f32 = 0.01;
const ACCELERATION: f32 = 13.0;
const JUMP_HEIGHT: f32 = 0.2 * PLAYER_HEIGHT;
const GRAVITY: f32 = 40.0 * PLAYER_HEIGHT;
const AIR_ACCELERATION: f32 = ACCELERATION / 2.0;
const PLAYER_HEIGHT: f32 = 0.01;

fn ray_plane(
    origin: glam::Vec3,
    direction: glam::Vec3,
    normal: glam::Vec3,
    center: glam::Vec3,
) -> f32 {
    let denom = normal.dot(direction);
    if f32::abs(denom) > 0.0001 {
        let t = (center - origin).dot(normal) / denom;
        if t >= 0.0 {
            t
        } else {
            f32::INFINITY
        }
    } else {
        f32::INFINITY
    }
}
const PLANE_NORMAL: glam::Vec3 = vec3(0.0, 1.0, 0.0);
const PLANE_POSITION: glam::Vec3 = vec3(0.0, 0.9999, 0.0);
const MAX_SLOPE_ANGLE: f32 = 55.0;
fn collide_and_slide(
    vel: glam::Vec3,
    org: glam::Vec3,
    world: &Vec<u32>,
    depth: u32,
    gizzmos: &mut GizzmoBuffer,
    gravity: bool,
    val_init: glam::Vec3,
) -> glam::Vec3 {
    if depth >= MAX_BOUNCES {
        return glam::Vec3::ZERO;
    }

    let max_dist = vel.length() + SKIN_WIDTH;

    let hit = ray_cast(world, org, vel.normalize(), max_dist);

    if let Some((hit_dist, normal)) = hit {
        let mut snap_to_surface = vel.normalize() * (hit_dist - SKIN_WIDTH);
        let mut leftover = vel - snap_to_surface;
        let angle = UP.angle_between(normal);

        if snap_to_surface.length() <= SKIN_WIDTH {
            snap_to_surface = glam::Vec3::ZERO;
        }
        if angle <= MAX_SLOPE_ANGLE {
            if gravity {
                return snap_to_surface;
            }
            let mag = leftover.length();
            leftover = leftover.project_onto_normalized(normal).normalize();
            leftover *= mag;
        } else {
            let scale = 1.0
                - vec3(normal.x, 0.0, normal.z)
                    .normalize()
                    .dot(-vec3(val_init.x, 0.0, val_init.z).normalize());
            let mag = leftover.length();
            leftover = leftover.project_onto_normalized(normal).normalize();
            leftover *= mag * scale;
        }

        if snap_to_surface.is_nan() {
            return glam::Vec3::ZERO;
        }

        return snap_to_surface
            + collide_and_slide(
                leftover,
                org + snap_to_surface,
                world,
                depth + 1,
                gizzmos,
                gravity,
                val_init,
            );
    }

    if vel.is_nan() {
        return glam::Vec3::ZERO;
    }

    return vel;
}

fn ray_cast(
    world: &Vec<u32>,
    org: glam::Vec3,
    dir: glam::Vec3,
    max_dist: f32,
) -> Option<(f32, glam::Vec3)> {
    let oct_tree_hit = ray_voxel(world, org, dir);
    let plane_hit = ray_plane(org, dir, PLANE_NORMAL, PLANE_POSITION);

    let hit = if let Some((pos, normal)) = oct_tree_hit {
        let dist = ((org - pos) as glam::Vec3).length();
        if dist < plane_hit {
            Some((dist, normal))
        } else if plane_hit != f32::INFINITY {
            Some((plane_hit, PLANE_NORMAL))
        } else {
            None
        }
    } else if plane_hit != f32::INFINITY {
        Some((plane_hit, PLANE_NORMAL))
    } else {
        None
    };
    if let Some((depth, _)) = hit {
        if depth <= max_dist {
            hit
        } else {
            None
        }
    } else {
        None
    }
}

const ligth_rotation: f32 = PI / 1.5;
const light_hight: f32 = PI / 4.0;

pub fn velocity(
    mut query: Query<(&mut Position, &mut PhysicsBody)>,
    time: Res<Time>,
    world: Res<GameWorld>,
    mut gizzmos: ResMut<GizzmoBuffer>,
) {
    let delta_time = time.delta_seconds();

    for (mut position, mut pb) in &mut query {
        let light_dir: glam::Vec3 = vec3(
            f32::sin(ligth_rotation) * f32::cos(light_hight),
            f32::sin(ligth_rotation) * f32::sin(light_hight),
            f32::cos(ligth_rotation),
        );
        let main_hit = ray_voxel(&world.build_tree, position.position, position.rotation);
        // if let Some(main_hit) = main_hit {
        //     let shadow_hit = ray_voxel(&world.build_tree, main_hit.0 + (main_hit.1 * 0.0001), light_dir);
        //     gizzmos.sphere(1, vec3(0.0, 1.0, 0.0), main_hit.0, 0.001);
        //     if let Some(shadow_hit) = shadow_hit {
        //         gizzmos.sphere(0, vec3(1.0, 0.0, 0.0), shadow_hit.0, 0.001);
        //     }else {
        //         gizzmos.sphere(0, vec3(1.0, 0.0, 0.0), glam::Vec3::ZERO, -1.0);
        //     }
        // }else {
        //     gizzmos.sphere(1, vec3(1.0, 0.0, 0.0), glam::Vec3::ZERO, -1.0);
        // }

        let center = vec3(
            position.position.x,
            position.position.y - PLAYER_HEIGHT / 2.0,
            position.position.z,
        );

        pb.grounded = ray_cast(&world.build_tree, center, vec3(0.0, -1.0, 0.0), 0.1).is_some();

        let gravity = vec3(0.0, -GRAVITY, 0.0);

        let mut vel_movement = pb.velocity.with_y(0.0);
        let mut vel_and_grav =
            pb.velocity.with_x(0.0).with_z(0.0) + gravity * delta_time * delta_time;

        vel_movement = collide_and_slide(
            vel_movement,
            center,
            &world.build_tree,
            0,
            &mut gizzmos,
            false,
            vel_movement,
        );
        vel_and_grav = collide_and_slide(
            vel_and_grav,
            center,
            &world.build_tree,
            0,
            &mut gizzmos,
            true,
            vel_and_grav,
        );
        pb.velocity = vel_movement + vel_and_grav;

        position.position += pb.velocity;

        if position.position == glam::Vec3::NAN {
            position.position = glam::Vec3::ZERO;
        }
    }
}

pub fn movement(
    mut query: Query<(&mut Position, &mut PhysicsBody), With<Player>>,
    controls: Res<Controls>,
    time: Res<Time>,
) {
    let delta_time = time.delta_seconds();
    for (mut position, mut pb) in &mut query {
        let side = position.rotation.cross(UP);

        // Update direction
        let new_direction = if controls.look_around {
            let side_rot =
                glam::Quat::from_axis_angle(side, -controls.cursor_delta[1] * ANGLE_PER_POINT);
            let y_rot = glam::Quat::from_rotation_y(-controls.cursor_delta[0] * ANGLE_PER_POINT);
            let rot = glam::Mat3::from_quat(side_rot * y_rot);

            (rot * position.rotation).normalize()
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
        let move_vec = direction * if pb.grounded { MOVE_SPEED } else { AIR_SPEED };
        pb.velocity.x = pb.velocity.x.lerp(
            move_vec.x * delta_time,
            delta_time
                * if pb.grounded {
                    ACCELERATION
                } else {
                    AIR_ACCELERATION
                },
        );
        pb.velocity.z = pb.velocity.z.lerp(
            move_vec.z * delta_time,
            delta_time
                * if pb.grounded {
                    ACCELERATION
                } else {
                    AIR_ACCELERATION
                },
        );
        if controls.go_up {
            pb.velocity.y += JUMP_HEIGHT;
        }
        position.rotation = new_direction;
    }
}
pub fn PlayerPlugin(app: &mut App) {
    app.add_systems(Update, (movement, velocity.after(movement)));
}
