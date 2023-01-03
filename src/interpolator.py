from tkinter import N
from unittest import result
import numpy as np
from dataio import Dataset
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

class Interpolater:
    def __init__(self, 
                interpolator="quadra-linear",
                focal_plane=1,
                aperture_size=1,
                undersampled=1.,):
        self.interpolator = interpolator
        assert interpolator in ["bilinear", "quadra-linear"]

        self.focal_plane = focal_plane
        assert focal_plane > 0 and focal_plane <= 1 # focal plane: 0~1s

        self.aperture_size = aperture_size
        self.undersampled = undersampled
    
    def interpolate_single_ray(self, lf, s, t, u, v):
        # it is a little bit hard, since s, t, u, v are all float
        # s, t, u, v: float
        S, T = lf.shape[0], lf.shape[1]
        # check pos is in the camera array
        if not(0 <= s < S and 0 <= t < T):
            return np.zeros(3) # outside our camera array
        # now the pos is in the camera array
        # and our aperture is set to be 1, i.e. aperture_size = 1
        # so we need to locate the four neighbor camera to interpolate
        ss, tt = int(s), int(t)
        wu, wv = s - ss, t - tt # interpolate weight
        cameras = [[ss, tt], [ss+1, tt], [ss, tt+1], [ss+1, tt+1]]
        weights = [(1-wu)*(1-wv), wu*(1-wv), (1-wu)*wv, wu*wv]
        # now we get the four neighbor camera and its corresponding weight
        result = np.zeros(3)
        sum_w = 0
        for camera, weight in zip(cameras, weights):
            result += weight * self.bilinear(lf, camera[0], camera[1], u, v) if camera[0] < S and camera[1] < T else np.zeros(3)
            sum_w += weight if camera[0] < S and camera[1] < T else 0
        return result / sum_w if sum_w !=0 else 1

    def interpolate(self, data: Dataset, pos: list):
        assert len(pos) == 2 # pos : position of virtual camera, [float, float]
        S, T = data.camera_shape
        U, V = data.image_shape

        self.basic_disparity = np.array([(self.focal_plane-1)/self.focal_plane, -(self.focal_plane-1)/self.focal_plane], dtype=np.float32)
        lf = data.data # light field: (s,t,u,v)
        ##padding
        lf, pos = self.padding(lf, pos)

        basic_camera = self.which_camera(pos) # left-top camera
        results = np.zeros((U, V, 3))

        assert type(self.aperture_size) == int and self.aperture_size >=1
        # if self.interpolator == "quadra-linear":
        if self.aperture_size == 1:
            cameras = [basic_camera, basic_camera+np.array([1,0]), basic_camera+np.array([0,1]), basic_camera+np.array([1,1])]
            disparities = [self.basic_disparity * (camera - pos) for camera in cameras] # caused from pos to cameras
            wu, wv = pos[0]-basic_camera[0], pos[1]-basic_camera[1]
            weights = [(1-wu)*(1-wv), wu*(1-wv), (1-wu)*wv, wu*wv]
            # print(weights)
        else: # implement Gaussian weight
            cameras = []
            disparities = []
            weights = []
            for i in range(-self.aperture_size + 1, self.aperture_size+1):
                for j in range(-self.aperture_size + 1, self.aperture_size+1):
                    new_camera = basic_camera + np.array([i,j])
                    cameras.append(new_camera)
                    new_disparity= self.basic_disparity * (new_camera - pos)
                    disparities.append(new_disparity)
                    weights.append(np.exp(-((np.linalg.norm(new_disparity)**2)/(2*self.aperture_size**2))))
            weights = np.array(weights) / np.sum(weights) if np.sum(weights) != 0 else np.zeros(len(weights))
            weights = weights.tolist()
        # else:
        #     assert self.aperture_size == 1
        #     cameras = [(basic_camera+np.array([0.5,0.5])).astype(int)]
        #     disparities = [self.basic_disparity * (camera - pos) for camera in cameras] # caused from pos to cameras
        #     weights = [1.0]

        cameras = [camera.tolist() for camera in cameras]
        disparities = [disparity.tolist() for disparity in disparities]

        vv, uu = np.meshgrid(np.linspace(0, V-1, V), np.linspace(0, U-1, U))
        uv_map = np.concatenate((uu.reshape(-1, 1), vv.reshape(-1, 1)), axis=1).astype(np.float32) # [U*V, 2] [[0,0], [0,1], [0,2], ...]
        total_weights = np.zeros(U*V) # mask shape
        # intepolate
        for camera, disparity, weight in zip(cameras, disparities, weights):
            ss, tt = int(camera[0]), int(camera[1])
            dispar_uv = uv_map + np.array(disparity, dtype=np.float32)
            dispar_u = dispar_uv[:, 0] #[U*V]
            dispar_v = dispar_uv[:, 1] #[U*V]
            # mask.shape = [U*V]
            mask = np.array(dispar_u<0, dtype=np.bool_) | np.array(dispar_u>=U, dtype=np.bool_) | np.array(dispar_v<0, dtype=np.bool_) | np.array(dispar_v>=V, dtype=np.bool_)
            dispar_uv[mask] = np.zeros(2) # anyway, we don't care about these results
            # weights = [(1-wu)*(1-wv), wu*(1-wv), (1-wu)*wv, wu*wv]
            camera_st = lf[ss][tt].reshape(-1, 3)

            if self.interpolator == "quadra-linear":
                dispar_uv = (dispar_uv).astype(int) #[U*V, 2]   
                wu, wv = dispar_u - dispar_uv[:, 0], dispar_v - dispar_uv[:, 1]
                wuv = np.concatenate(((1-wu)*(1-wv), wu*(1-wv), (1-wu)*wv, wu*wv), axis=0) #[U*V*4]
                # order -> left_top, left_down, right_top, right_down
                uv = np.concatenate((dispar_uv, dispar_uv + np.array([1,0], dtype=int), dispar_uv + np.array([0,1], dtype=int), dispar_uv + np.array([1,1], dtype=int)), axis=0)
                uv = uv[:,0] * (V+1) + uv[:,1] # very hard to debug since we pad the image
                col = camera_st[uv]
                weighted_col = np.sum((wuv[:, None] * col).reshape(4, -1, 3), axis=0, keepdims=False)

            elif self.interpolator == "bilinear":
                uv = dispar_uv + np.array([0.5, 0.5])
                uv = (uv).astype(int) #[U*V, 2]   nearest pixel
                uv = uv[:,0] * (V+1) + uv[:,1] # very hard to debug since we pad the image
                weighted_col = camera_st[uv]

            weighted_col[mask] = np.zeros(3)
            total_weights += weight
            total_weights[mask] -= weight
            results += weight * weighted_col.reshape(U, V, 3)
        # avoid division by zero
        total_weights = total_weights.reshape(U, V)
        total_weights[total_weights == 0] = 1
        results /= total_weights.reshape(U, V, 1) # normalize
        results[total_weights == 0] = np.zeros(3)
        # for u in range(U):
        #     for v in range(V):
        #         sum_w = 0
        #         for camera, disparity, weight in zip(cameras, disparities, weights):
        #             ss, tt = int(camera[0]), int(camera[1])
        #             uu, vv = u + disparity[0], v + disparity[1]
        #             results[u, v] += weight * uv_interpolator(lf, ss, tt, uu, vv) if 0<=ss<S and 0<=tt<T and 0<=uu<U and 0<=vv<V else np.zeros(3)
        #             sum_w += weight if 0<=ss<S and 0<=tt<T and 0<=uu<U and 0<=vv<V else 0
        #         results[u, v] /= sum_w if sum_w != 0 else 1
        if 0 < self.undersampled < 1:
            new_U, new_V = int(self.undersampled * U), int(self.undersampled * V)
            assert U % new_U ==0 and V % new_V == 0
            t_U, t_V = int(U / new_U), int(V / new_V)
            # undersampled results 
            undersampled_results = results[0:U:t_U, 0:V:t_V]
            results = undersampled_results
        return results

    def bilinear(self, lf, s, t, u, v):
        u0, v0 = int(u), int(v)
        wu, wv = u-u0, v-v0
        p0 = lf[s,t,u0,v0]
        p1 = lf[s,t,u0+1,v0] if u0+1<lf.shape[2] else np.zeros(3)
        p2 = lf[s,t,u0,v0+1] if v0+1<lf.shape[3] else np.zeros(3)
        p3 = lf[s,t,u0+1,v0+1] if u0+1<lf.shape[2] and v0+1<lf.shape[3] else np.zeros(3)
        sum_w = (1-wu)*(1-wv)
        sum_w += wu*(1-wv) if u0+1<lf.shape[2] else 0
        sum_w += (1-wu)*wv if v0+1<lf.shape[3] else 0
        sum_w += wu*wv if u0+1<lf.shape[2] and v0+1<lf.shape[3] else 0
        return ((1-wu)*(1-wv)*p0 + wu*(1-wv)*p1 + wv*(1-wu)*p2 + wu*wv*p3) / sum_w  

    def coarse_approximate(self, lf, s, t, u, v):
        return lf[s,t,int(u),int(v)]

    def which_camera(self, pos: np.ndarray):
        # pos: [float, float]
        # return: [int, int]
        return np.array([int(pos[0]), int(pos[1])], dtype=np.float32)


    def padding(self, lf: np.ndarray, pos: list):
        # print("Extend the light field by aperture_size to avoid boundary problem")
        before_pad = lf.shape
        lf = np.pad(lf, ((self.aperture_size, self.aperture_size), (self.aperture_size, self.aperture_size), (0, 1), (0, 1), (0,0)), 'edge')
        after_pad = lf.shape
        # print("Light field shape before padding: {before_pad}, after padding: {after_pad}".format(before_pad=before_pad, after_pad=after_pad))
        pos = np.array([pos[0] + self.aperture_size, pos[1] + self.aperture_size], dtype=np.float32)

        return lf, pos


class Camera:
    def __init__(self, 
                depth=0.,
                focal_plane=1./9.,
                H=240.,
                W=320.):
        self.depth = depth
        self.focal_plane = focal_plane
        self.H = H
        self.W = W
        disparity = (1-self.focal_plane) / self.focal_plane
        dis_c = disparity * focal_plane / (1 - focal_plane) # neighbor camera distance (pixels)
        self.disparity = disparity
        # assert -15*dis_c*focal_plane/max(H, W) <= self.depth < self.focal_plane, "Depth must be in [-15*dis_c*focal_plane/max(H, W), focal_plane)"
        self.dis_c = dis_c

        if self.depth >=0: # moving forward
            self.new_W = (self.W - 1) * (self.focal_plane - self.depth) / self.focal_plane
            self.new_H = (self.H - 1) * (self.focal_plane - self.depth) / self.focal_plane
            self.start_u, self.start_v = (self.H - 1 - self.new_H) / 2, (self.W - 1 - self.new_W) / 2
            self.end_u, self.end_v = self.start_u + self.new_H, self.start_v + self.new_W
            vv, uu = np.meshgrid(np.linspace(self.start_v, self.end_v, self.W), np.linspace(self.start_u, self.end_u, self.H))
            uv_map = np.concatenate((uu.reshape(-1, 1), vv.reshape(-1, 1)), axis=1).astype(np.float32)
            
            # u, v.size() = [H*W]
            u, v = uv_map[:, 0], uv_map[:, 1]
            uv = uv_map.astype(int) #[H*W, 2]
            wu, wv = u - uv[:, 0], v - uv[:, 1]
            self.wuv = np.concatenate(((1-wu)*(1-wv), wu*(1-wv), (1-wu)*wv, wu*wv), axis=0) #[H*W*4]
            # order -> left_top, left_down, right_top, right_down
            self.uv = np.concatenate((uv, uv + np.array([1, 0], dtype=int), uv + np.array([0, 1], dtype=int), uv + np.array([1, 1], dtype=int)), axis=0) #[U*V*4, 2]
            self.uv = self.uv [:, 0] * (W+1) + self.uv[:, 1]
        else: # moving backward
            self.new_W = int(self.W * (-depth + focal_plane) / focal_plane)
            self.new_H = int(self.H * (-depth + focal_plane) / focal_plane)

    def get_col(self,  data:Dataset):
        interpolater = Interpolater(focal_plane=1./9., aperture_size=1)

        if self.depth >=0: #moving forward
            # padding _results to avoid boundary problem
            results = interpolater.interpolate(data, [7.5, 7.5])
            _results = np.pad(results, ((0, 1), (0, 1), (0, 0)), 'edge')
            results_line = _results.reshape(-1, 3)
            col = results_line[self.uv]
            weighted_col = np.sum((self.wuv[:, None] * col).reshape(4, -1, 3), axis=0, keepdims=False)
            return weighted_col.reshape(self.H, self.W, 3)
        else: #moving backward
            self.new_H = int(self.H * (1 - self.depth))
            self.new_W = int(self.W * (1 - self.depth))
            results = np.zeros((self.new_H, self.new_W, 3))
            self.start_u, self.start_v = int((self.new_H - self.H) / 2), int((self.new_W - self.W) / 2)
            self.end_u, self.end_v = self.start_u + self.H, self.start_v + self.W

            
            # center of the new image can be seen by center camera
            results[self.start_u:self.end_u, self.start_v:self.end_v] = interpolater.interpolate(data, [7.5, 7.5])[:, ::-1, :]
            # the rest part can not be seen by center camera, but can be seen by other cameras
            # left top square
            for i in range(0, self.start_u):
                for j in range(0, self.start_v):
                    dis_u = i - self.start_u
                    dis_v = j - self.start_v
                    s = 7.5 + dis_u / self.disparity
                    t = 7.5 + dis_v / self.disparity
                    results[i, j] = interpolater.interpolate_single_ray(data.data[:,:,:,::-1,:], s, t, 0, 0)
            # mid top  
            for i in range(0, self.start_u):
                for j in range(self.start_v, self.start_v + self.W):
                    dis_u = i - self.start_u
                    dis_v = 0
                    s = 7.5 + dis_u / self.disparity
                    t = 7.5 + dis_v / self.disparity
                    results[i, j] = interpolater.interpolate_single_ray(data.data[:,:,:,::-1,:], s, t, 0, j - self.start_v)

            # right top square
            for i in range(0, self.start_u):
                for j in range(self.start_v + self.W, self.new_W):
                    dis_u = i - self.start_u
                    dis_v = j - self.start_v - self.W
                    s = 7.5 + dis_u / self.disparity
                    t = 7.5 + dis_v / self.disparity
                    results[i, j] = interpolater.interpolate_single_ray(data.data[:,:,:,::-1,:], s, t, 0, self.W-1)

            # left mid
            for i in range(self.start_u, self.start_u + self.H):
                for j in range(0, self.start_v):
                    dis_u = 0
                    dis_v = j - self.start_v
                    s = 7.5 + dis_u / self.disparity
                    t = 7.5 + dis_v / self.disparity
                    results[i, j] = interpolater.interpolate_single_ray(data.data[:,:,:,::-1,:], s, t, i - self.start_u, 0)
            
            # right mid
            for i in range(self.start_u, self.start_u + self.H):
                for j in range(self.start_v + self.W, self.new_W):
                    dis_u = 0
                    dis_v = j - self.start_v - self.W
                    s = 7.5 + dis_u / self.disparity
                    t = 7.5 + dis_v / self.disparity
                    results[i, j] = interpolater.interpolate_single_ray(data.data[:,:,:,::-1,:], s, t, i - self.start_u, self.W-1)
            
            # left down square
            for i in range(self.start_u + self.H, self.new_H):
                for j in range(0, self.start_v):
                    dis_u = i - self.start_u - self.H
                    dis_v = j - self.start_v
                    s = 7.5 + dis_u / self.disparity
                    t = 7.5 + dis_v / self.disparity
                    results[i, j] = interpolater.interpolate_single_ray(data.data[:,:,:,::-1,:], s, t, self.H-1, 0)
            
            # mid down
            for i in range(self.start_u + self.H, self.new_H):
                for j in range(self.start_v, self.start_v + self.W):
                    dis_u = i - self.start_u - self.H
                    dis_v = 0
                    s = 7.5 + dis_u / self.disparity
                    t = 7.5 + dis_v / self.disparity
                    results[i, j] = interpolater.interpolate_single_ray(data.data[:,:,:,::-1,:], s, t, self.H-1, j - self.start_v)

            # right down square
            for i in range(self.start_u + self.H, self.new_H):
                for j in range(self.start_v + self.W, self.new_W):
                    dis_u = i - self.start_u - self.H
                    dis_v = j - self.start_v - self.W
                    s = 7.5 + dis_u / self.disparity
                    t = 7.5 + dis_v / self.disparity
                    results[i, j] = interpolater.interpolate_single_ray(data.data[:,:,:,::-1,:], s, t, self.H-1, self.W-1)


            results = cv2.resize(results[:, ::-1, :], (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return results

            #----------------------------------------------------------------------------------
            # S = self.H * (-self.depth) / self.focal_plane / self.dis_c
            # T = self.W * (-self.depth) / self.focal_plane / self.dis_c
            # results = np.zeros((self.H, self.W, 3))
            # for i in range(self.H):
            #     for j in range(self.W):
            #         s = 7.5 - S / 2. + i / (self.H - 1) * S
            #         t = 7.5 - T / 2. + j / (self.W - 1) * T
            #         results[i, j] = interpolater.interpolate_single_ray(data.data[:, :, :, ::-1, :], s, t, i, j)
            # return results[:, ::-1, :]
            #----------------------------------------------------------------------------------
            # results.size() = (self.new_H, self.new_W, 3)
            # we should use several view to compose the results
            # results = np.zeros((self.new_H, self.new_W, 3), dtype=np.float32)
            # weights = np.zeros((self.new_H, self.new_W), dtype=np.float32)
            # # We sample a 4*4 virtual camera array on the projection plane onto camera array
            # # and use the virtual camera array to compose the results
            # S = self.H * (-self.depth) / self.focal_plane / self.dis_c
            # T = self.W * (-self.depth) / self.focal_plane / self.dis_c
            # camera_s = np.linspace(0, S, 2, endpoint=True)
            # camera_t = np.linspace(0, T, 2, endpoint=True)
            # for s in camera_s:
            #     for t in camera_t:
            #         u = s / S * (self.new_H - self.H)
            #         v = t / T * (self.new_W - self.W)
            #         result = interpolater.interpolate(data, [7.5+s-S/2., 7.5+t-T/2.])[:, ::-1, :]
            #         ul = int(min(self.H-1, self.new_H - u))
            #         vl = int(min(self.W-1, self.new_W - v))
            #         # since (u,v) is not interger point, thus we need to interpolate the result
            #         lt_u = int(u)
            #         lt_v = int(v)
            #         print(u, ul, u+ul)
            #         print(v, vl, v+vl)
            #         wu, wv = u - lt_u, v - lt_v
            #         wuv = np.array([(1-wu)*(1-wv), wu*(1-wv), (1-wu)*wv, wu*wv])
            #         uv = np.array([[lt_u, lt_v], [lt_u+1, lt_v], [lt_u, lt_v+1], [lt_u+1, lt_v+1]])
            #         results[lt_u:lt_u+ul, lt_v:lt_v+vl] += result[:ul, :vl] * wuv[0]
            #         results[lt_u+1:lt_u+1+ul, lt_v:lt_v+vl] += result[:ul, :vl] * wuv[1]
            #         results[lt_u:lt_u+ul, lt_v+1:lt_v+1+vl] += result[:ul, :vl] * wuv[2]
            #         results[lt_u+1:lt_u+1+ul, lt_v+1:lt_v+1+vl] += result[:ul, :vl] * wuv[3]
            #         weights[lt_u:lt_u+ul, lt_v:lt_v+vl] += wuv[0]
            #         weights[lt_u+1:lt_u+1+ul, lt_v:lt_v+vl] += wuv[1]
            #         weights[lt_u:lt_u+ul, lt_v+1:lt_v+1+vl] += wuv[2]
            #         weights[lt_u+1:lt_u+1+ul, lt_v+1:lt_v+1+vl] += wuv[3]   

            #     results = results / weights[:, :, None]
            #     # avoid division by zero
            #     results[np.isnan(results)] = 0
            # # undersampled the results to the original size by interpolating
            # # results.size() = (self.H, self.W, 3)
            # results = cv2.resize(results[:, ::-1, :], (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            # return results
            #----------------------------------------------------------------------------------

def variable_focal_plane(data: Dataset, interpolator="quadra-linear", aperture_size=1):
    focal_planes = np.arange(0.05, 0.25, 0.0001)
    results = []
    pbar = tqdm(focal_planes)
    for focal_plane in pbar:
        pbar.set_description("Prcoessing focal plane: {focal_plane:.3f}".format(focal_plane=focal_plane))
        interpolater = Interpolater(focal_plane=focal_plane, interpolator=interpolator, aperture_size=aperture_size)
        results.append(interpolater.interpolate(data, [7.5, 7.5]))

    path = "results/variable_focal_plane_ap{aperture_size:d}.avi".format(aperture_size=aperture_size)
    generate_video(results, path, fps=60)

def variable_virtual_camera(data: Dataset, interpolator="quadra-linear", aperture_size=1):
    results = []
    S, T = data.camera_shape
    # traverse all virtual cameras first line
    var_t =  np.linspace(0, T, 32, endpoint=False)
    var_s =  np.linspace(0, S, 32, endpoint=False)
    interpolater = Interpolater(interpolator=interpolator, aperture_size=aperture_size, focal_plane=1./9.)
    for s in var_s:
        pbar = tqdm(var_t)  
        for t in pbar:
            pbar.set_description("Prcoessing virtual camera:({s:.2f}, {t:.2f})".format(s=s, t=t))
            results.append(interpolater.interpolate(data, [s, t]))
    
    path = "results/" + interpolator + ".avi"
    generate_video(results, path, fps=32)

def variable_aperture_size(data: Dataset, interpolator="quadra-linear"):
    results = []
    aperture_sizes = np.linspace(1, 8, 8, endpoint=True)
    pbar = tqdm(aperture_sizes)
    for aperture_size in pbar:
        pbar.set_description("Prcoessing aperture size: {aperture_size:2d}".format(aperture_size=int(aperture_size)))
        interpolater = Interpolater(interpolator=interpolator, aperture_size=int(aperture_size), focal_plane=1./9.)
        results.append(interpolater.interpolate(data, [7.5, 7.5]))
    
    path = "results/variable_aperture_size_only.avi"
    generate_video(results, path, fps=2)

def unsampled_light_field(data: Dataset, interpolator="bilinear"):

    for undersampled in [0.5, 0.25, 0.125, 1.0]:
        results = []
        S, T = data.camera_shape
        # traverse all virtual cameras first line
        var_t =  np.linspace(0, T, 32, endpoint=False)
        var_s =  np.linspace(0, S, 32, endpoint=False)
        interpolater = Interpolater(interpolator=interpolator, focal_plane=1./9., undersampled = undersampled)
        for s in var_s:
            pbar = tqdm(var_t)  
            for t in pbar:
                pbar.set_description("Prcoessing virtual camera:({s:.2f}, {t:.2f})".format(s=s, t=t))
                results.append(interpolater.interpolate(data, [s, t]))
        
        path = "results/{interpolator:s}_unsampled_{undersampled:.3f}.avi".format(interpolator=interpolator, undersampled=undersampled)
        generate_video(results, path, fps=32)

def expand_field_of_view(data:Dataset, focal_plane=1./9.,interpolator="quadra-linear"):
    camera_depth = np.linspace(0, focal_plane, 240, endpoint=False)
    results = []
    pbar = tqdm(camera_depth)
    for depth in pbar:
        pbar.set_description("Prcoessing camera depth: {depth:.3f}".format(depth=depth))
        camera = Camera(depth, focal_plane, data.image_shape[0], data.image_shape[1])
        image_depth = camera.get_col(data)
        results.append(image_depth)
    path = "results/expand_field_of_view_2.avi"
    generate_video(results, path, fps=30)

def generate_video(results, path, fps=60):
    results = np.array(results)
    results = np.array(results, dtype=np.uint8)
    results = results[:, :, :, ::-1]
    # save video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (results.shape[2], results.shape[1]))
    for i in range(results.shape[0]):
        out.write(results[i])
    out.release()

def task(data:Dataset, task:  str, interpolator: str):
    if task == "interpolation":
        base_dir = "results/interpolation"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if interpolator == "bilinear":
            #Task1.1 bilinear interpolation [done]
            interpolater = Interpolater(interpolator="bilinear", focal_plane=1./9.)
            result = interpolater.interpolate(data, [7.9, 7.9])
            generate_image(result, "results/interpolation/bilinear.png")
        elif interpolator == "quadra-linear":
            #Task1.2 quadra-linear interpolation [done]
            interpolater = Interpolater(interpolator="quadra-linear", focal_plane=1./9.)
            result = interpolater.interpolate(data, [7.9, 7.9])
            generate_image(result, "results/interpolation/quadra-linear.png")
        variable_virtual_camera(data, interpolator)

    elif task == "undersampled":
        base_dir = "results/undersampled"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if interpolator == "bilinear":
            #Task 1.3.1: unsampled light field using bilinear interpolation [done]
            for undersampled in [1.0, 0.5, 0.25, 0.125]:  
                interpolater = Interpolater(interpolator="bilinear", focal_plane=1./9., undersampled=undersampled)
                result = interpolater.interpolate(data, [7.5, 7.5])
                generate_image(result, "results/undersampled/bilinear_unsampled_bilinear_{undersampled:.3f}.png".format(undersampled=undersampled))
        elif interpolator == "quadra-linear":
            #Task 1.3.2: unsampled light field using quadra-linear interpolation [done]
            for undersampled in [1.0, 0.5, 0.25, 0.125]:  
                interpolater = Interpolater(interpolator="quadra-linear", focal_plane=1./9., undersampled=undersampled)
                result = interpolater.interpolate(data, [7.5, 7.5])
                generate_image(result, "results/undersampled/quadra-linear_unsampled_quadra-linear_{undersampled:.3f}.png".format(undersampled=undersampled))
        unsampled_light_field(data, interpolator)

    elif task == "variable_focal_plane":
        # Task 2: variable_focal_plane [done]
        base_dir = "results/focal_plane"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        focal_planes = np.arange(0.25, 1.0, 0.01)
        for focal_plane in focal_planes:
            interpolater = Interpolater(interpolator="bilinear", focal_plane=focal_plane, aperture_size=8)
            result = interpolater.interpolate(data, [7.5, 7.5])
            generate_image(result, "results/focal_plane/focal_plane_{focal_plane:.4f}.png".format(focal_plane=focal_plane))
        variable_focal_plane(data, "quadra-linear", aperture_size=8)

    elif task =="variable_aperture_size":
        # Task 3: variable_aperture_size [done]
        base_dir = "results/aperture_size"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        aperture_sizes = np.linspace(1, 8, 8, endpoint=True)
        for _aperture_size in aperture_sizes:
            aperture_size = int(_aperture_size)
            interpolater = Interpolater(interpolator="quadra-linear", focal_plane=0.2, aperture_size=aperture_size)
            result = interpolater.interpolate(data, [7.5, 7.5])
            generate_image(result, "results/aperture_size/aperture_size_{aperture_size:1d}_2.png".format(aperture_size=aperture_size))
        variable_aperture_size(data)

    elif task =="expand_field_of_view":
        # Task 4: expand_field_of_view
        base_dir = "results/expand_field_of_view"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        # camera_depth = np.linspace(-0.06, 0, 120, endpoint=False)
        camera_depth = np.linspace(-2.0, 0.11, 150, endpoint=False)
        # camera_depth = np.concatenate((camera_depth, np.linspace(0, 0.06, 300, endpoint=False)), axis=0)
        pbar = tqdm(camera_depth)
        results = []
        for depth in pbar:
            pbar.set_description("Prcoessing camera depth: {depth:.4f}".format(depth=depth))
            camera = Camera(depth, 1./9., data.image_shape[0], data.image_shape[1])
            image_depth = camera.get_col(data)
            results.append(image_depth)
            generate_image(image_depth, "results/expand_field_of_view/depth_{depth:.4f}.png".format(depth=depth))
        path = "results/expand_field_of_view.avi"
        generate_video(results, path, fps=30)

def generate_image(results, path):
    results = np.array(results)
    results = np.array(results, dtype=np.uint8)
    results = results[:, :, ::-1]
    cv2.imwrite(path, results)

if "__main__" == __name__:
    pass
    # path = 'data/'
    # data = Dataset(path)
    # data.show_all()

    # results_dir = "results"
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)

    #Task 1: Interpolater 
    #Task 1.1: bilinear interpolation [done]
    # variable_virtual_camera(data, "bilinear")
    #Task 1.2: quadra-linear interpolation [done]
    # variable_virtual_camera(data, "quadra-linear")
    #Task 1.3: unsampled light field using bilinear interpolation [done]
    # unsampled_light_field(data, "bilinear")
    # unsampled_light_field(data, "quadra-linear")

    #Task 2: variable_focal_plane [done]
    # variable_focal_plane(data, "quadra-linear", aperture_size=8)

    #Task 3: variable_aperture_size [done]
    # variable_aperture_size(data)

    #Task 4: expand_field_of_view [done]
    # expand_field_of_view(data)


    
