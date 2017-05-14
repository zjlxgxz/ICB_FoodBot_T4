# coding=utf-8
import MySQLdb

class SearchDB:
	#DB connection init
	def __init__(self ,host ,username ,passwd ,dbname):
		try:
			self.db = MySQLdb.connect(host ,username ,passwd ,dbname ,charset='utf8')
		except MySQLdb.Error as e:
			print "Error %d: %s" % (e.args[0], e.args[1])

	def grabData(self ,intent ,slots):	#slots is a dictionary
		print 'action : ' + str(intent)
		print 'slots : ' + str(slots) 
		try:
			cursor = self.db.cursor()
			tmp = []
			restname = []
			i=0
			
			if slots['RESTAURANTNAME'] != '':
				tmp = slots['RESTAURANTNAME'].split(' ')
				if tmp[-1].__contains__('?') or tmp[-1].__contains__('.'):
					tmp[-1] = ''
					if tmp.__len__() > 3:
						tmp[-2] = ''
					for i in range(tmp.__len__()):
						if tmp[i] != '':
							restname.append(tmp[i])
					slots['RESTAURANTNAME'] = ' '.join(restname)
			print slots['RESTAURANTNAME']
			
			if intent == 'Get_Restaurant' :
				sql_query = 'SELECT * FROM restaurant WHERE categories LIKE %s and displayAddress LIKE %s' %('\'%'+slots['CATEGORY']+'%\'' ,'\'%'+slots['LOCATION']+'%\'')
				cursor.execute(sql_query)
				#print sql_query
			elif intent == 'Get_location' and slots['RESTAURANTNAME'] != '' :
				cursor.execute('SELECT displayAddress FROM restaurant WHERE name LIKE %s LIMIT 1' %('\'%'+slots['RESTAURANTNAME']+'%\''))

			elif intent == 'Get_rating' and slots['RESTAURANTNAME'] != '' :
				cursor.execute('SELECT rating FROM restaurant WHERE name LIKE %s LIMIT 1' %('\'%'+slots['RESTAURANTNAME']+'%\''))

			#elif intent == 'Get_comment' and slots['RESTAURANTNAME'] != '' :
			#	pass
			else :
				pass

			results = cursor.fetchall()
			#print 'results : ' + str(results)
			content = ''
			if results.__len__() == 0:
				pass
			else:
				if intent == 'Get_Restaurant' :
					content = {'RESTAURANTNAME':str(results[slots['TIMES']][0]) ,'LOCATION':results[slots['TIMES']][5]}
				else:
					for record in results:
						if intent == 'Get_location' :
							content = {'LOCATION':str(record[0])}
						elif intent == 'Get_rating' :
							content = {'RATING':str(record[0])+'/5.0 stars'}
			#	elif intent == 'Get_comment' :
			#		pass
			self.db.close()
			print content
			return content
		except MySQLdb.Error as e:
			return ''

if __name__ == '__main__':

	slots = {'CATEGORY':'' ,'RESTAURANTNAME':'2nd city here?' ,'LOCATION':'adfasdf' ,'TIME':'' ,'TIMES':2}
	search = SearchDB('140.112.49.151' ,'foodbot' ,'welovevivian' ,'foodbotDB')
	search.grabData('Get_Restaurant' ,slots)
