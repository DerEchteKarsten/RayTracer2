use bevy::{app::DynEq, prelude::*};
use glam::{Vec4Swizzles, *};

use crate::{components::{PhysicsBody, Player, Position}, oct_tree::{ray_voxel, GameWorld, Octant}, render_system::GizzmoBuffer, Camera, CameraUniformData, Controls};
const MOVE_SPEED: f32 = 10.0;
const ANGLE_PER_POINT: f32 = 0.0009;
const UP: glam::Vec3 = vec3(0.0, 1.0, 0.0);
const AIR_SPEED: f32 = 0.02;
const MAX_BOUNCES: u32 = 5;
const SKIN_WIDTH: f32 = 0.015;

fn project_to_plane(vec: glam::Vec3, normal: glam::Vec3) -> glam::Vec3 {
    let dist = vec.dot(normal);
    vec - dist*normal
}

fn ray_plane(origin: glam::Vec3, direction: glam::Vec3, normal: glam::Vec3, center: glam::Vec3) -> f32{
    let denom = normal.dot(direction);
    if f32::abs(denom) > 0.0001
    {
        let t = (center - origin).dot(normal) / denom;
        if t >= 0.0 {
            t
        }else {
            f32::INFINITY
        }
    }else {
        f32::INFINITY
    }
}
const PLANE_NORMAL: glam::Vec3 = vec3(0.0, 1.0, 0.0);
const PLANE_POSITION: glam::Vec3 = vec3(0.0, 0.9999, 0.0);
const MAX_SLOPE_ANGLE: f32 = 55.0;
fn collide_and_slide(vel: glam::Vec3, org: glam::Vec3, world: &Vec<u32>, depth: u32, gizzmos: &mut GizzmoBuffer, gravity: bool, val_init: glam::Vec3) -> glam::Vec3 {
    if depth >= MAX_BOUNCES {
        return glam::Vec3::ZERO;
    }
    
    let max_dist = vel.length() + SKIN_WIDTH;
    let dir = vel.normalize();

    let hit = ray_cast(world, org, dir, max_dist);
    
    if let Some((pos, normal)) = hit {
        let hit_dist = (pos-org).length();
        // gizzmos.draw_gizzmo(0, normal, org + dir * hit_dist, 0.01);
        let mut snap_to_surface = dir * (hit_dist - SKIN_WIDTH);
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
        }else {
            let scale = 1.0 - vec3(normal.x, 0.0, normal.z).normalize().dot(-vec3(val_init.x, 0.0, val_init.z).normalize());
            let mag = leftover.length();
            leftover = leftover.project_onto_normalized(normal).normalize();
            leftover *= mag * scale;
        }
        info!("{}", leftover);
        return snap_to_surface + collide_and_slide(leftover, pos + snap_to_surface, world, depth+1, gizzmos, gravity, val_init);
    }
    return vel;
}

fn ray_cast(world: &Vec<u32>, org: glam::Vec3, dir: glam::Vec3, max_dist: f32) -> Option<(glam::Vec3, glam::Vec3)> {
    let oct_tree_hit = ray_voxel(world, org, dir);
    let plane_hit = ray_plane(org, dir, PLANE_NORMAL, PLANE_POSITION);

    let hit = if let Some((pos, normal)) = oct_tree_hit {
        if ((org-pos) as glam::Vec3).length() < plane_hit {
            Some((pos,normal))
        }else if plane_hit != f32::INFINITY {
            Some((org + dir * plane_hit, PLANE_NORMAL))
        }else {
            None
        }
    }else if plane_hit != f32::INFINITY {
        Some((org + dir * plane_hit, PLANE_NORMAL))
    }else {
        None
    };
    if let Some((pos, _)) = hit {
        if (org-pos).length() <= max_dist {
            hit
        } else {
            None
        }
    } else {
        None
    }
}

pub fn velocity(mut query: Query<(&mut Position, &mut PhysicsBody)>, time: Res<Time>, world: Res<GameWorld>, mut gizzmos: ResMut<GizzmoBuffer>) {
    let delta_time = time.delta_seconds();
    for (mut position, mut vel) in &mut query {
        let feet = vec3(position.position.x, position.position.y, position.position.z);
        vel.grounded = ray_cast(&world.build_tree, feet, vec3(0.0, -1.0, 0.0), 0.1).is_some();
        
        vel.velocity = collide_and_slide(vel.velocity, feet, &world.build_tree, 0, &mut gizzmos, false, vel.velocity);
        
        if !vel.grounded {
            let gravity = vec3(0.0, -10.0, 0.0) * delta_time;
            vel.velocity = collide_and_slide(gravity, feet, &world.build_tree, 0, &mut gizzmos, true, gravity);
        }else {
            vel.velocity.y = 0.0;
        }
         
        position.position += vel.velocity * delta_time;
        
        if position.position == glam::Vec3::NAN {
            position.position = glam::Vec3::ZERO;
        }
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
    app
    .add_systems(Update, movement)
    .add_systems(PostUpdate, velocity);
}