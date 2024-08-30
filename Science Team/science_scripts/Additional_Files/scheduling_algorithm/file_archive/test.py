from observatory_characteristics import observatoryList
from observatory_characteristics import Observatory
from priority_calculator import PriorityCalculator
from satellitetracker import SatelliteTracker
import time
from datetime import datetime
from datetime import date
import numpy

def main():
    pc = PriorityCalculator()
    #for item in observatoryList:
    #    print(item.name + str(pc.computePriority(item)))
    st = SatelliteTracker()
    t = datetime(2024, 9, 6)
    alt = st.computeAltitude(observatoryList[0], t)
    for obs in observatoryList:
        pv = pc.computePriority(obs, t)
        print(obs.name + " " + str(pv))
        

if __name__ == "__main__":
    main()
     