from volsurfs import OccupancyGrid
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.geometry.primitives.bounding_box import BoundingBox


def init_occupancy_grid(bounding_primitive, res=256):
    scene_radius = bounding_primitive.get_radius()
    occupancy_grid = OccupancyGrid(
        res, [scene_radius * 2, scene_radius * 2, scene_radius * 2]
    )
    if isinstance(bounding_primitive, BoundingSphere):
        occupancy_grid.init_sphere_roi(scene_radius, 0.0)
    return occupancy_grid
