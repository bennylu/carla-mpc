import carla
import config as Config
import numpy as np
import math


class PyGameDrawer():

    def __init__(self, main):
        self.main = main
        self.pygame = main.game.pygame
        self.camera = main.game.camera
        self.font_14 = self.pygame.freetype.SysFont('Times New Roman', 14)

    # draw on the camera perspective

    def __w_locs_2_camera_locs(self, w_locs):
        camera_locs = []
        for w_loc in w_locs:
            bbox = PyGameDrawer.get_location_bbox(w_loc, self.camera)
            if math.isnan(bbox[0, 0]) or math.isnan(bbox[0, 1]):
                camera_locs.append((-1, -1))
            camera_locs.append((int(bbox[0, 0]), int(bbox[0, 1])))
        return camera_locs

    def draw_camera_text(self, location, color, text):
        x, y = self.__w_locs_2_camera_locs([location])[0]
        if x >= 0 and x <= Config.PYGAME_WIDTH and y >= 0 and y <= Config.PYGAME_HEIGHT:
            self.font_14.render_to(self.main.surface, (x, y), text, color)

    def draw_camera_circles(self, w_locs, color, radius):
        cam_locs = self.__w_locs_2_camera_locs(w_locs)
        for cam_loc in cam_locs:
            self.pygame.draw.circle(
                self.main.surface, color, cam_loc, radius, 1)

    def draw_camera_polygon(self, w_locs, color):
        if len(w_locs) < 3:
            return
        points = self.__w_locs_2_camera_locs(w_locs)
        self.pygame.draw.polygon(self.main.surface, color, points, 4)

    def draw_camera_lines(self, color, w_locs, width=1):
        cam_locs = self.__w_locs_2_camera_locs(w_locs)
        for i in range(len(cam_locs) - 1):
            self.__draw_camera_line_safe(color, [cam_locs[i][0], cam_locs[i][1]], [
                cam_locs[i + 1][0], cam_locs[i + 1][1]], width)

    def __draw_camera_line_safe(self, color, pt1, pt2, width=1):
        if (pt1[0] >= 0 and pt1[0] <= Config.PYGAME_WIDTH and pt1[1] >= 0 and pt1[1] <= Config.PYGAME_HEIGHT and pt2[0] >= 0 and pt2[0] <= Config.PYGAME_WIDTH and pt2[1] >= 0 and pt2[1] <= Config.PYGAME_HEIGHT):
            self.pygame.draw.line(self.main.surface, color, pt1, pt2, width)

    @staticmethod
    def get_location_bbox(location, camera):
        bb_cords = np.array([[0, 0, 0, 1]])
        cords_x_y_z = PyGameDrawer.location_to_sensor_cords(
            bb_cords, location, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def location_to_sensor_cords(cords, location, sensor):
        world_cord = PyGameDrawer.location_to_world_cords(cords, location)
        sensor_cord = PyGameDrawer._world_to_sensor_cords(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def location_to_world_cords(cords, location):
        bb_transform = carla.Transform(location)
        vehicle_world_matrix = PyGameDrawer.get_matrix(bb_transform)
        world_cords = np.dot(vehicle_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _create_vehicle_bbox_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """
        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor_cords(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """
        world_cord = PyGameDrawer._vehicle_to_world_cords(cords, vehicle)
        sensor_cord = PyGameDrawer._world_to_sensor_cords(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world_cords(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = PyGameDrawer.get_matrix(bb_transform)
        vehicle_world_matrix = PyGameDrawer.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor_cords(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """
        sensor_world_matrix = PyGameDrawer.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix
