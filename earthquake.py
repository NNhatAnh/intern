from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core import Stream
import os
import time

client = Client("IRIS")

start_time = UTCDateTime("2000-01-01T00:00:00")
end_time = UTCDateTime.now()

min_latitude = 20.0 
max_latitude = 46.0 
min_longitude = 128.0
max_longitude = 149.0

min_magnitude = 6.5

pre_event_seconds = 120
post_event_seconds = 600

output_dir = "waveform_japan"
os.makedirs(output_dir, exist_ok=True)

print(f"Đang tìm kiếm các trận động đất lớn (magnitude >= {min_magnitude}) ở Nhật Bản từ {start_time.date} đến {end_time.date}...")

try:
    events = client.get_events(
        starttime=start_time,
        endtime=end_time,
        minmagnitude=min_magnitude,
        maxlatitude=max_latitude,
        minlatitude=min_latitude,
        maxlongitude=max_longitude,
        minlongitude=min_longitude,
        orderby="magnitude"
    )
    print(f"Tìm thấy {len(events)} trận động đất lớn.")

    if not events:
        print("Không tìm thấy trận động đất nào thỏa mãn tiêu chí tìm kiếm.")
        print("Vui lòng thử điều chỉnh các tham số tìm kiếm (ví dụ: giảm min_magnitude).")

    for i, event in enumerate(events):
        event_time = event.origins[0].time
        event_latitude = event.origins[0].latitude
        event_longitude = event.origins[0].longitude
    
        event_magnitude = event.magnitudes[0].mag if event.magnitudes else "N/A"
    
        event_id = event.resource_id.id.split('=')[-1]

        print(f"\nĐang xử lý sự kiện {i+1}/{len(events)}: ID {event_id} (Magnitude: {event_magnitude}, Thời gian: {event_time})")

        waveform_starttime = event_time - pre_event_seconds
        waveform_endtime = event_time + post_event_seconds
        station_search_radius_deg = 5
        event_stream = Stream()

        try:
            inventory = client.get_stations(
                starttime=waveform_starttime,
                endtime=waveform_endtime,
                latitude=event_latitude,
                longitude=event_longitude,
                maxradius=station_search_radius_deg,
                channel="BH*",
                level="response"
            )
            
            num_stations_found = sum(len(net.stations) for net in inventory.networks)
            print(f"Tìm thấy {num_stations_found} trạm hoạt động trong bán kính {station_search_radius_deg} độ.")

            if num_stations_found == 0:
                print(f"Không tìm thấy trạm nào có kênh BH* hoạt động trong cửa sổ thời gian và bán kính cho sự kiện {event_id}.")
                print("Bỏ qua sự kiện này và chuyển sang sự kiện tiếp theo.")
                continue
        
            for network in inventory:
                for station in network:
                    try:
                    
                        st = client.get_waveforms(
                            network=network.code,
                            station=station.code,
                            location="*",
                            channel="BH*",
                            starttime=waveform_starttime,
                            endtime=waveform_endtime
                        )
                        if st:
                            event_stream += st
                            print(f"  Đã tải {len(st)} trace từ trạm {station.code} (Mạng lưới: {network.code}).")
                        else:
                            print(f"  Không có dữ liệu sóng cho trạm {station.code} (Mạng lưới: {network.code}) trong cửa sổ thời gian.")

                    except Exception as e_waveform:
                        print(f"  Lỗi khi tải dữ liệu sóng từ {network.code}.{station.code} cho sự kiện {event_id}: {e_waveform}")
                    
                    
                    time.sleep(0.5)
            
        
            if event_stream:
            
                event_stream.merge(fill_value='interpolate', method=1)
                
            
            
                print("  Đang loại bỏ đáp ứng thiết bị...")
                try:
                
                
                    event_stream.remove_response(inventory=inventory, output='VEL') 
                    print("  Đã loại bỏ đáp ứng thiết bị thành công.")
                except Exception as e_remove_response:
                    print(f"  Lỗi khi loại bỏ đáp ứng thiết bị cho sự kiện {event_id}: {e_remove_response}")
                    print("  Dữ liệu sẽ được lưu dưới dạng thô (counts) nếu không thể loại bỏ đáp ứng.")
    
                output_filepath = os.path.join(output_dir, f"{event_id}.mseed")  
                event_stream.write(output_filepath, format="MSEED")
                print(f"Đã tải, xử lý và lưu tổng cộng {len(event_stream)} trace vào {output_filepath}")
            else:
                print(f"Không có dữ liệu sóng nào được tải về cho sự kiện {event_id} từ tất cả các trạm đã tìm thấy.")

        except Exception as e_station:
            print(f"Lỗi khi tìm kiếm trạm hoặc xử lý dữ liệu cho sự kiện {event_id}: {e_station}")
            print("Có thể do không có trạm nào trong bán kính, hoặc lỗi mạng/API. Bỏ qua sự kiện này.")
            continue

        time.sleep(1)

except Exception as e_main:
    print(f"Lỗi chính khi tìm kiếm sự kiện động đất: {e_main}")
    print("Vui lòng kiểm tra kết nối internet hoặc thử lại sau.")

print("\nHoàn tất quá trình tải dữ liệu.")