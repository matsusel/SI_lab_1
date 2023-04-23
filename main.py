import datetime
import pandas as pd
from datetime import time, timedelta

pd.set_option('display.max_rows', None, 'display.max_columns', None)

filename = 'C:\\Users\\Suseł\\Downloads\\connection_graph.csv'

def load_data():
    df = pd.read_csv(filename)
    df['departure_time'] = df['departure_time'].apply(adjust_time)
    df['arrival_time'] = df['arrival_time'].apply(adjust_time)
    return df

# (['Unnamed: 0', 'company', 'line', 'departure_time', 'arrival_time',
#        'start_stop', 'end_stop', 'start_stop_lat', 'start_stop_lon',
#        'end_stop_lat', 'end_stop_lon'],
#       dtype='object')


def adjust_time(x):
    h = int(x[0:2]) % 24
    m = int(x[3:5])
    s = int(x[6:])
    return datetime.time(hour=h, minute=m, second=s)


# Node = { 'start_stop' : Edges[[company, line,arrival_time,  departure time, end_stop, end_stop_lat, end_stop_lon]] ,}

def create_graph(data):
    node = {}
    stops = data['start_stop'].unique()
    # print(stops)
    for stop in stops:
        for index, s in data[data['start_stop'] == stop].iterrows():
            if s['start_stop'] not in node:
                node[s['start_stop']] = []
            node[s['start_stop']].append([s['line'], s['end_stop'], s['departure_time'], s['arrival_time'],
                                          s['end_stop_lat'], s['end_stop_lon'], s['start_stop_lat'],
                                          s['start_stop_lon']])
    for key in node.keys():
        node[key].sort(key=lambda x: x[2])

    return node
    # [0] line
    # [1] end_stop
    # [2] departure_time Czas wyjazdu
    # [3] arrival_time Czas przybycia
    # [4] end_stop_lat
    # [5] end_stop_lon
    # [6] start_stop_lat
    # [7] start_stop_lon


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        self.elements.append((item, priority))
        self.elements.sort(key=lambda x: x[1])

    def get(self):
        return self.elements.pop(0)[0]


def print_path_lines(path):
    for item in path:
        print("Wyjazd: " + str(item[3]) + " z " + str(item[0]) + " Przyjazd: " + str(item[2]) + " do " + str(item[4]
            ) + " Linia:" + str(item[1]))


def dijkstra_search(graph, start, end, current_time):
    queue = PriorityQueue()
    queue.put(start, 0)

    cost = {node: float('inf') for node in graph}
    cost[start] = 0
    previous = {node: None for node in graph}

    while not queue.empty():
        current = queue.get()

        if current == end:
            path = []
            node = end
            while node != start:
                path.append(previous[node])
                node = previous[node][0]
            return path[::-1]

        for neighbor in graph[current]:
            if current != start:
                current_time = previous[current][4]
            neighbor_line, neighbor_stop, departure_time, arrival_time, _, _, _, _ = neighbor
            if current_time <= arrival_time and departure_time >= current_time:
                neighbor_cost = cost[current] + (arrival_time.hour * 60 + arrival_time.minute - current_time.hour * 60
                                                 - current_time.minute)
                if neighbor_cost < cost[neighbor_stop]:
                    cost[neighbor_stop] = neighbor_cost
                    previous[neighbor_stop] = (current, neighbor_line, neighbor_stop, departure_time, arrival_time)
                    queue.put(neighbor_stop, neighbor_cost)

    return []


def heuristic_cost(node1_lat, node1_lon, node2_lat, node2_lon):
    return abs(node1_lat - node2_lat) + abs(node1_lon - node2_lon)


def a_star_time(graph, start, end, current_time):
    queue = PriorityQueue()
    queue.put(start, 0)

    cost = {node: float('inf') for node in graph}
    cost[start] = 0
    previous = {node: None for node in graph}

    while not queue.empty():
        current = queue.get()

        if current == end:
            path = []
            node = end
            while node != start:
                path.append(previous[node])
                node = previous[node][0]
            return path[::-1]

        for neighbor in graph[current]:
            neighbor_line, neighbor_stop, departure_time, arrival_time, neighbor_lat, neighbor_lon, start_lat,\
                start_lon = neighbor
            if current != start:
                current_time = previous[current][4]
            if current_time <= arrival_time and departure_time >= current_time:
                neighbor_cost = cost[current] + (
                        arrival_time.hour * 60 + arrival_time.minute - current_time.hour * 60 - current_time.minute) \
                                + heuristic_cost(neighbor_lat, neighbor_lon, start_lat, start_lon)

                if neighbor_cost < cost[neighbor_stop]:
                    cost[neighbor_stop] = neighbor_cost
                    previous[neighbor_stop] = (current, neighbor_line, neighbor_stop, departure_time, arrival_time)
                    queue.put(neighbor_stop, neighbor_cost)

    return []


def a_star_time_better(graph, start: str, end: str, current_time: datetime.time):
    queue = PriorityQueue()
    queue.put(start, 0)

    cost = {node: float('inf') for node in graph}
    cost[start] = 0
    previous = {node: None for node in graph}

    while not queue.empty():
        current = queue.get()

        if current == end:
            path = []
            node = end
            while node != start:
                path.append(previous[node])
                node = previous[node][0]
            return path[::-1]

        for neighbor in graph[current]:
            neighbor_line, neighbor_stop, departure_time, arrival_time, neighbor_lat, neighbor_lon, start_lat, \
                start_lon = neighbor
            if current != start:
                current_time = previous[current][4]
            if current_time <= arrival_time and departure_time >= current_time:
                neighbor_cost = cost[current] + (
                        arrival_time.hour * 60 + arrival_time.minute - current_time.hour * 60 - current_time.minute) \
                                + heuristic_cost(neighbor_lat, neighbor_lon, start_lat, start_lon)

                if neighbor_cost < cost[neighbor_stop]:
                    cost[neighbor_stop] = neighbor_cost
                    previous[neighbor_stop] = (current, neighbor_line, neighbor_stop, departure_time, arrival_time)
                    queue.put(neighbor_stop, neighbor_cost)

    return []


def a_star_line(graph, start, end, current_time):
    queue = PriorityQueue()
    queue.put(start, 0)

    cost = {node: float('inf') for node in graph}
    cost[start] = 0
    previous = {node: None for node in graph}

    while not queue.empty():
        current = queue.get()
        if current == end:
            path = []
            node = end
            while node != start:
                path.append(previous[node])
                node = previous[node][0]
            return path[::-1]

        for neighbor in graph[current]:
            if current != start:
                current_time = previous[current][4]
                d_line = previous[current][1]

            neighbor_line, neighbor_stop, departure_time, arrival_time, neighbor_lat, neighbor_lon, start_lat, \
                start_lon = neighbor
            if current != start:
                current_time = previous[current][4]
            if current_time <= arrival_time and departure_time >= current_time:
                neighbor_cost = cost[current] + (heuristic_cost(neighbor_lat, neighbor_lon, start_lat, start_lon))
                if current != start:
                    if (d_line != neighbor_line):
                        neighbor_cost += 10000

                if neighbor_cost < cost[neighbor_stop]:
                    cost[neighbor_stop] = neighbor_cost
                    previous[neighbor_stop] = (current, neighbor_line, neighbor_stop, departure_time, arrival_time)
                    queue.put(neighbor_stop, neighbor_cost)

    return []


if __name__ == '__main__':
     data = load_data()
     graph = create_graph(data)
     start_node = 'DWORZEC AUTOBUSOWY'
     start = datetime.datetime.now()
     for i in range(100):
        path1 = dijkstra_search(graph, start_node, 'most Grunwaldzki', datetime.time(10, 0, 0))
     end = datetime.datetime.now()
     print(str(end - start))
     start = datetime.datetime.now()
     for i in range(100):
         path2 = a_star_time(graph, start_node, 'most Grunwaldzki', datetime.time(10, 0, 0))
     end = datetime.datetime.now()
     print(str(end - start))
     start = datetime.datetime.now()
     for i in range(100):
         path3 = a_star_time_better(graph, start_node, 'most Grunwaldzki', datetime.time(10, 0, 0))
     end = datetime.datetime.now()
     print(str(end - start))
     start = datetime.datetime.now()
     for i in range(100):
         path4 = a_star_line(graph, start_node, 'most Grunwaldzki', datetime.time(10, 0, 0))
     end = datetime.datetime.now()
     print(str(end - start))
     print_path_lines(path1)
     print_path_lines(path2)
     print_path_lines(path3)
     print_path_lines(path4)

