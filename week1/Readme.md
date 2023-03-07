Assignment 1:
Bước 1: Cài đặt thư viện geopandas

Bước 2: git clone https://github.com/CityScope/CSL_HCMC
Bước 3: dùng geopandas để đọc shapefile trong /Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp

Bước 4: hãy thực hiện các tác vụ truy vấn sau
- Phường nào có diện tích lớn nhất
- Phường nào có dân số 2019 (Pop_2019) cao nhất
- Phường nào có diện tích nhỏ nhất
- Phường nào có dân số thấp nhất (2019)
- Phường nào có tốc độ tăng trưởng dân số nhanh nhất (dựa trên Pop_2009 và Pop_2019)
- Phường nào có tốc độ tăng trưởng dân số thấp nhất
- Phường nào có biến động dân số nhanh nhất
- Phường nào có biến động dân số chậm nhất
- Phường nào có mật độ dân số cao nhất (2019)
- Phường nào có mật độ dân số thấp nhất (2019)
///---------------------------------------------------------///
Assignment 2:
Bước 1: Cài đặt geopandas và folium

Bước 2: git clone https://github.com/CityScope/CSL_HCMC
Bước 3: dùng geopandas để đọc shapefile trong /Data/GIS/Population/population_HCMC/population_shapefile/Population_District_Level.shp

Bước 4: hãy thực hiện vẽ ranh giới các quận lên bản đồ dựa theo hướng dẫn sau:
https://geopandas.readthedocs.io/en/latest/gallery/polygon_plotting_with_folium.html
///---------------------------------------------------------///
Assignment 3:
Bước 1: Cài đặt các thư viện cần thiết

!pip install matplotlib==3.1.3
!pip install osmnet
!pip install folium

!pip install rtree
!pip install pygeos
!pip install geojson
!pip install geopandas

Bước 2: clone data từ https://github.com/CityScope/CSL_HCMC

Bước 3: Load ranh giới quận huyện và dân số quận huyện từ: Data\GIS\Population\population_HCMC\population_shapefile\Population_District_Level.shp

Bước 4: Load dữ liệu click của người dùng

Bước 5: Lọc ra 10 quận huyện có tốc độ tăng dân số nhanh nhất (Pop2019/Pop2017)

Bước 6: Dùng spatial join (from geopandas.tools import sjoin) để lọc ra các điểm click của người dùng trong 5 quận/huyện hot nhất

Bước 7: chạy KMean cho top 10 quận huyện này. Lấy K = 20

Bước 8: Lưu cụm điểm nhiều nhất của mỗi quận huyện

Bước 9: show lên bản đồ các cụm đông nhất theo từng quận huyện theo dạng HEATMAP

Bước 10: Lưu heatmap xuống file png
