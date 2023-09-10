import sys, math
#sys.argv
#busID, lineID, coordX, coordY, time

class point:
    def __init__(self, coordinates, timestamp):
        self.coordinates = coordinates
        self.timestamp = timestamp
    
def main():
    fileName = sys.argv[1]
    flag = sys.argv[2]
    option = sys.argv[3]
    
    if flag == '-b':
        totDistance = calc_tot_distance(fileName, option)
        print('Total Distance run by ', option, ' is: ', totDistance)
    elif flag == '-l':
        avgSpeed = calc_avg_speed(fileName, option)
        print('Average spped on line ', option, ' is: ', avgSpeed)

    return 0

def calc_tot_distance(fileName, busID):
    distance = 0
    path = []
    with open(fileName, 'r') as file:
        for line in file:
            fields = line.split(' ')
            busIDread = fields[0]
            x = int(fields[2])
            y = int(fields[3])
            if busIDread == busID:
                path.append(point((x, y), 0))

        for i in range(len(path) - 1):
            # sqrt((x1-x2)^2 + (y1-y2)^2)
            distance += math.sqrt((path[i].coordinates[0]-path[i + 1].coordinates[0])**2 + (path[i].coordinates[1]-path[i + 1].coordinates[1])**2)

    return distance

def calc_avg_speed(fileName, lineID):
    distance = {} #{busId : distance}
    time = {} #{busId : time}
    path = {} #{busId : listOfPoints}
    totDistance = 0
    totTime = 0
    with open(fileName, 'r') as file:
        for line in file:
            busIdread, lineIdread, x, y, timestamp = line.split(' ')[0:5]

            if lineIdread == lineID:
                if not busIdread in path.keys():
                    path[busIdread] = []
                path[busIdread].append(point((x, y), timestamp))
                
        for key, values in path.items():
            for i in range(len(values) - 1):
                if not key in distance.keys():
                    distance[key] = 0
                if not key in time.keys():
                    time[key] = 0    
                # sqrt((x1-x2)^2 + (y1-y2)^2)
                distance[key] += math.sqrt((int(values[i].coordinates[0])-int(values[i + 1].coordinates[0]))**2 + (int(values[i].coordinates[1])-int(values[i + 1].coordinates[1]))**2)
                time[key] += abs(int(values[i + 1].timestamp) - int(values[i].timestamp))

            totDistance += distance[key]
            totTime += time[key]

    return totDistance/totTime

main()