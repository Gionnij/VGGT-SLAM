import open3d as o3d

pcd = o3d.io.read_point_cloud("/Users/giovannichiementin/Desktop/Thesis/data/poses_points_livestream.pcd")
o3d.visualization.draw_geometries([pcd])




