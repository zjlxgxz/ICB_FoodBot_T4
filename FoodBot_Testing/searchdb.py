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

			if intent == 'Get_Restaurant' :
				if slots['CATEGORY'] != '' and slots['LOCATION'] != '' :
					cursor.execute('SELECT * FROM restaurant WHERE categories LIKE %s and displayAddress LIKE %s LIMIT 1' %('\'%'+slots['CATEGORY']+'%\'' ,'\'%'+slots['LOCATION']+'%\''))
				elif slots['CATEGORY'] != '' :
					cursor.execute('SELECT * FROM restaurant WHERE categories LIKE %s LIMIT 1' %('\'%'+slots['CATEGORY']+'%\''))
				elif slots['LOCATION'] != '' :
					cursor.execute('SELECT * FROM restaurant WHERE displayAddress LIKE %s LIMIT 1' %('\'%'+slots['LOCATION']+'%\''))

			elif intent == 'Get_location' and slots['RESTAURANTNAME'] != '' :
				cursor.execute('SELECT displayAddress FROM restaurant WHERE name LIKE %s LIMIT 1' %('\'%'+slots['RESTAURANTNAME']+'%\''))

			elif intent == 'Get_rating' and slots['RESTAURANTNAME'] != '' :
				cursor.execute('SELECT rating FROM restaurant WHERE name LIKE %s LIMIT 1' %('\'%'+slots['RESTAURANTNAME']+'%\''))

			#elif intent == 'Get_comment' and slots['RESTAURANTNAME'] != '' :
			#	pass
			else :
				print 'match nothing'

			results = cursor.fetchall()
			print 'results : ' + str(results)
			for record in results:
				if intent == 'Get_Restaurant' :
					print 'I recommend ' + str(record[0]) + ' on ' + str(record[5])
				elif intent == 'Get_location' :
					print 'It\'s located on ' + str(record[0])
				elif intent == 'Get_rating' :
					print str(record[0]) + '/5.0 stars'
			#	elif intent == 'Get_comment' :
			#		pass
			self.db.close()
		except MySQLdb.Error as e:
			print "Error %d: %s" % (e.args[0], e.args[1])

if __name__ == '__main__':

	slots = {'CATEGORY':'' ,'RESTAURANTNAME':'' ,'LOCATION':'' ,'TIME':''}
	search = SearchDB('140.112.49.151' ,'foodbot' ,'welovevivian' ,'foodbotDB')
	
	slots['RESTAURANTNAME'] = 'the'
	search.grabData('Get_rating' ,slots)
