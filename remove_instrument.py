from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.inventory import Inventory
import os

output_folder = "waveform_output"
os.makedirs(output_folder, exist_ok=True)

client = Client("IRIS")
cnt = 0
error = 0

for fname in os.listdir("waveforms"):
    cnt+=1
    st = read(os.path.join("waveforms", fname))
    tr = st[0]

    print(tr.id)
    print(tr.stats.starttime, tr.stats.endtime)

    start = tr.stats.starttime
    end   = tr.stats.endtime

    try:
        inv = client.get_stations(network=tr.stats.network,
                                station=tr.stats.station,
                                location=tr.stats.location,
                                channel=tr.stats.channel,
                                starttime=start,
                                endtime=end,
                                level="response")

        pre_filt = (0.01, 0.02, 45.0, 50.0)
        st.remove_response(inventory=inv, pre_filt=pre_filt, output="ACC", water_level=60, zero_mean=True)

        st.write(f"{output_folder}/{fname}", format="MSEED")
    except:
        print("Cant find data")
        error += 1
    
    finally:
        print(f"There are {cnt} files")
        print(f"Error {error} files")
