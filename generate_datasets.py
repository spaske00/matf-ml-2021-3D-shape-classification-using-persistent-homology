import numpy as np
from pathlib import Path


def make_point_clouds(n_samples_per_shape: int, n_points: int, noise: float):
    """Make point clouds for circles, spheres, and tori with random noise.
    """
    circle_point_clouds = [
        np.random.uniform(1, 10, 1)[0] *np.asarray(
            [
                [np.sin(t) + noise * (np.random.rand(1)[0] - 0.5), np.cos(t) + noise * (np.random.rand(1)[0] - 0.5), 0]
                for t in range((n_points ** 2))
            ]
        )
        for kk in range(n_samples_per_shape)
    ]
    

    sphere_point_clouds = [
        np.random.uniform(1, 10, 1)[0] * np.asarray(
            [
                [
                    np.cos(s) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.cos(s) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for kk in range(n_samples_per_shape)
    ]
    
    R = 8 # Distance from the center of the tube to the center of the torus
    r = 2 # radis of the tube
    noise += r/R + np.log(R/r)
    def generate_torus(R, r, noise, n_points):
        return np.asarray(
            [
                [
                    (R + r*np.cos(s)) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    (R + r*np.cos(s)) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    0.1*r*np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in np.linspace(0, 2*np.pi, int(1.5*n_points))
                for s in np.linspace(0, 2*np.pi, int(1.5*n_points))
            ]
        )
    
    torus_point_clouds = [
        generate_torus(R,r,noise, n_points)
        for kk in range(n_samples_per_shape)
    ]
    
   
    
    double_torus_point_clouds = []
    for index, torus in enumerate(torus_point_clouds):
        shifted_torus = np.copy(torus)
        shifted_torus = generate_torus(R * 0.6, r * 0.6, noise, n_points)
        shifted_torus.T[0] += 1.5*R
        shifted_torus.T[1] += 1.5*R
        double_torus_point_clouds.append(
            np.concatenate([
                torus,
                shifted_torus
            ])
        )
    
    
    point_clouds = {
        'circle': circle_point_clouds,
        'sphere': sphere_point_clouds,
        'torus': torus_point_clouds,
        'double_torus':double_torus_point_clouds
    }
    
    return point_clouds
