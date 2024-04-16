import math

import numpy as np


def calculate_distance_and_angle(cords_first, cords_second):
    """Calculates the distance and angle between two objects with x, y coordinates.

    Args:
        x1: x-coordinate of the first object.
        y1: y-coordinate of the first object.
        x2: x-coordinate of the second object.
        y2: y-coordinate of the second object.

    Returns:
        A tuple containing the distance and angle (in degrees) between the two objects.
    """
    x1, y1 = cords_first
    x2, y2 = cords_second
    # Calculate the distance
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Calculate the angle in radians
    angle = math.atan2(y2 - y1, x2 - x1)

    # Convert angle to degrees
    angle_deg = math.degrees(angle)

    return distance, angle_deg


def distance_to_all_rectangle_walls(point, corner1, corner2):
    """Calculates the distance to all walls of a rectangle from a point.

    Args:
        point: A tuple representing the coordinates (x, y) of the point.
        corner1: A tuple representing the coordinates (x1, y1) of one corner of the rectangle.
        corner2: A tuple representing the coordinates (x2, y2) of the opposite corner of the rectangle.

    Returns:
        A list containing the distances to all four walls of the rectangle.

    Raises:
        ValueError: If the provided points do not define a valid rectangle.
    """

    # Check if points define a valid rectangle (opposite corners)
    if corner1[0] == corner2[0] and corner1[1] == corner2[1]:
        raise ValueError("Points do not define a valid rectangle (same coordinates)")

    point = np.array(point)
    corner1 = np.array(corner1)
    corner2 = np.array(corner2)

    point_matrix = np.tile(point, (2, 1))

    distances = np.abs(point_matrix - np.array([corner1, corner2]))

    return distances


def calculate_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx**2 + dy**2)
