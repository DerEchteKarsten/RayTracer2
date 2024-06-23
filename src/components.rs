use bevy::prelude::*;
use glam::Vec3;
#[derive(Component)]
pub struct Position {
    pub position: Vec3,
    pub rotation: Vec3,
}

#[derive(Component)]
pub struct Transform {
    pub scale: Vec3,
}

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct PhysicsBody {
    pub velocity: Vec3,
    pub grounded: bool,
}
