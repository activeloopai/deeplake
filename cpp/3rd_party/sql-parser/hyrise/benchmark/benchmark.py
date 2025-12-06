#!/bin/python
from __future__ import print_function

import urllib, urllib2
import json

class HyriseConnection(object):
	def __init__(self, host, port):
		super(HyriseConnection, self).__init__()
		self.host = host
		self.port = port

	def __hyriseurl(self):
		return 'http://%s:%d/query' % (self.host, self.port)

	def __parseResponse(self, response):
		result = json.loads(response)
		if 'error' in result:
			print('An error occurred: %s' % result['error'][0])
			return None

		if 'performanceData' in result:
			pf_data = result['performanceData']
			total_time = 0
			parse_time = 0
			querytask_time = 0

			for operator in pf_data:
				time_ms = operator['endTime'] - operator['startTime']
				total_time += time_ms
				if operator['name'] == 'RequestParseTask':
					parse_time += time_ms
				if operator['name'] == 'SQLQueryTask':
					querytask_time += time_ms

			return {
				'total_ms': total_time,
				'parse_ms': parse_time,
				'querytask_ms': querytask_time,
				'preparation_ms': parse_time + querytask_time
			}

	def __sendRequest(self, params):
		url = self.__hyriseurl()
		params['performance'] = 'true'
		data = urllib.urlencode(params)
		req = urllib2.Request(url, data)
		try:
			rsp = urllib2.urlopen(req)
			response = rsp.read();
			return self.__parseResponse(response)
		except TypeError as e:
			print("An error occurred")
			return None
		except Exception as e:
			return self.__parseResponse(e.read())

	def __aggregatePerfArray(self, perfArray):
		perf = perfArray[0]
		for data in perfArray[1:]:
			for key in data:
				perf[key] += data[key]

		for key in perf:
			perf[key] /= len(perfArray)

		return perf

	def executeSingleSQL(self, sql):
		params = {'sql': sql}
		return self.__sendRequest(params)

	def executeSingleJSON(self, jsonString):
		params = {'query': jsonString}
		return self.__sendRequest(params)

	def executeSQL(self, sql, times=1):
		perf = [self.executeSingleSQL(sql) for _ in range(times)]
		return self.__aggregatePerfArray(perf)

	def executeJSON(self, json, times=1):
		perf = [self.executeSingleJSON(json) for _ in range(times)]
		return self.__aggregatePerfArray(perf)
		


queries = {
	'select-1': {
		'sql': "SELECT name, city FROM students WHERE grade <= 2.0",
		'json': """{"operators":{"0":{"type":"GetTable","name":"students"},"1":{"type":"SimpleTableScan","predicates":[{"type":"LTE_V","in":0,"f":"grade","value":2,"vtype":1}]},"2":{"type":"ProjectionScan","fields":["name","city"]}},"edges":[["0","1"],["1","2"]]}""",

		'prepare': "PREPARE sel_test: SELECT name, city FROM students WHERE grade <= ?",
		'execute': "EXECUTE sel_test(2.0);"

	},

	'insert-1': {
		'sql': "INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);",
		'json': """{"operators":{"0":{"type":"GetTable","name":"students"},"1":{"type":"InsertScan","data":[["Max",42,"Musterhausen",2.3]]},"commit":{"type":"Commit"}},"edges":[["0","1"],["1","commit"]]}"""
	},

	'insert-2': {
		'sql': """
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
		""",
		'json': """{
	        "operators": {
	            "0": {
	                "type": "GetTable",
	                "name": "students"
	            },
	            "1": {
	                "type": "InsertScan",
	                "data": [
	                    ["Max", 42, "Musterhausen", 2.3],
	                    ["Max", 42, "Musterhausen", 2.3],
	                    ["Max", 42, "Musterhausen", 2.3],
	                    ["Max", 42, "Musterhausen", 2.3],
	                    ["Max", 42, "Musterhausen", 2.3],
	                    ["Max", 42, "Musterhausen", 2.3],
	                    ["Max", 42, "Musterhausen", 2.3],
	                    ["Max", 42, "Musterhausen", 2.3]
	                ]
	            },
		        "commit" : {
		            "type" : "Commit"
		        }
	        },
	        "edges": [["0","1"],["1","commit"]]
	    }""",
	    'prepare': """PREPARE batch_insert {
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
			INSERT INTO students VALUES ('Max', 42, 'Musterhausen', 2.3);
	    }""",
	    'execute': "EXECUTE batch_insert;"
	}
}

if __name__ == '__main__':
	hyrise = HyriseConnection('localhost', 5000)

	# Load Table
	hyrise.executeSQL("CREATE TABLE IF NOT EXISTS students FROM TBL FILE 'test/students.tbl';")

	query = queries['insert-2']

	times = 50


	# if 'prepare' in query:	hyrise.executeSQL(query['prepare'])
	
	if 'sql' in query: 		print('SQL: ', hyrise.executeSQL(query['sql'], times))

	if 'execute' in query: 	print('Prepared: ', hyrise.executeSQL(query['execute'], times))

	# if 'json' in query:		print('JSON: ', hyrise.executeJSON(query['json'], times))
