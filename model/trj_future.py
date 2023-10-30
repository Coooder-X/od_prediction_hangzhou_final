import pyproj

def latlon_to_xy(latitude, longitude):
    # 定义墨卡托投影坐标系
    proj_merc = pyproj.Proj(proj='merc', ellps='WGS84')

    # 将经纬度转换为墨卡托投影坐标
    x, y = proj_merc(longitude, latitude)

    return x, y

# 示例用法
latitude = 40.7128
longitude = -74.0060

x, y = latlon_to_xy(latitude, longitude)
print(f"经纬度 ({latitude}, {longitude}) 转换为墨卡托坐标 ({x}, {y})")
