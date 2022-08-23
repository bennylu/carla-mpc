#!/usr/bin/env python3

import carla
import config as Config
import weakref
import numpy as np
import pygame


class SyncPyGame():
    def __init__(self, main):
        self.main = main
        self.camera = None
        self.display = None
        self.image = None
        self.capture = True

        self.pygame = pygame
        pygame.init()

        self.set_synchronous_mode(True)
        self.setup_camera()
        self.display = pygame.display.set_mode(
            (Config.PYGAME_WIDTH, Config.PYGAME_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.main.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 1 / Config.PYGAME_FPS
        settings.no_rendering_mode = True
        settings.actor_active_distance = 100
        self.main.world.apply_settings(settings)

    def setup_camera(self):
        camera_bp = self.main.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(Config.PYGAME_WIDTH))
        camera_bp.set_attribute('image_size_y', str(Config.PYGAME_HEIGHT))
        camera_bp.set_attribute('fov', str(Config.PYGAME_FOV))

        camera_transform = carla.Transform(
            carla.Location(x=-12, z=10), carla.Rotation(pitch=-30))

        self.camera = self.main.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.main.ego)
        weak_self = weakref.ref(self)
        self.camera.listen(
            lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = Config.PYGAME_WIDTH / 2.0
        calibration[1, 2] = Config.PYGAME_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = Config.PYGAME_WIDTH / \
            (2.0 * np.tan(Config.PYGAME_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    @staticmethod
    def set_image(weak_self, img):
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def get_surface(self):
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype('uint8'))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            return pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def game_loop(self, world, on_tick):
        try:
            while True:
                world.tick()
                pygame.event.get()

                self.capture = True
                surface = self.get_surface()
                if surface is not None:
                    self.main.surface = surface
                    on_tick()
                    self.display.blit(surface, (0, 0))
                    pygame.display.flip()

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.main.ego.destroy()
            pygame.quit()
