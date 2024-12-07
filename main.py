import torch
if __name__ == '__main__':
    # 接下来移除一些Gaussians，它们满足下列要求中的一个：
    max_screen_size = 20
    max_radii2D = torch.tensor([12, 2, 21], device="cuda")
    scaling = torch.tensor([[12, 12, 12], [2, 2, 2], [21, 21, 21]], device="cuda")
    opacity = torch.tensor([[-2.1], [-9], [7]], device="cuda")
    extent = 6.9
    min_opacity = 0.005
    prune_mask = (opacity < min_opacity).squeeze()  # 1. 接近透明（不透明度小于min_opacity）
    print("prune_mask_old", prune_mask)
    big_points_vs = max_radii2D > max_screen_size  # 2. 在某个相机视野里出现过的最大2D半径大于屏幕（像平面）大小
    print("big_points_vs", big_points_vs)
    big_points_ws = scaling.max(dim=1).values > 0.1 * extent  # 3. 在某个方向的最大缩放大于0.1 * extent（也就是说很长的长条形也是会被移除的）
    print("big_points_ws", big_points_ws)
    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    print("prune_mask_new", prune_mask)





