# coding=utf-8
import MySQLdb
import random

class SearchDB:
	#DB connection init
	def __init__(self ,host ,username ,passwd ,dbname):
		try:
			self.db = MySQLdb.connect(host ,username ,passwd ,dbname ,charset='utf8')
		except MySQLdb.Error as e:
			print "Error %d: %s" % (e.args[0], e.args[1])

	def grabData(self ,intent ,slots):	#slots is a dictionary
		try:
			cursor = self.db.cursor()
			tmp = []
			restname = []
			i = 0
			j = 0
			name = ''
			content = ''
			
			if slots['restaurant_name'] != '':
				tmp = slots['restaurant_name'].split(' ')
				if tmp.__len__() >= 4:
					j = tmp.__len__()/2-1
					for i in range(3):
						restname.append(tmp[j])
						j += 1
					name = ' '.join(restname)
				else:
					name = slots['restaurant_name']
			print name

			if slots['category'] == '':
				slots['category'] = ''
			if slots['area'] == '':
				slots['area'] = ''
			if slots['price'] == '':
				price = ''
			elif float(slots['price'].replace('$','')) < 10:
				price = 'Under$10'
			elif float(slots['price'].replace('$','')) > 60:
				price = 'Above$61'
			elif float(slots['price'].replace('$','')) >= 10 and float(slots['price'].replace('$','')) <= 30:
				price = '$11-30'
			elif float(slots['price'].replace('$','')) >= 30 and float(slots['price'].replace('$','')) <= 60:
				price = '$31-60'
			if slots['score'] == '':
				slots['score'] = '0.0'

			if intent == 'inform_restaurant':
				print '------'
				print slots
				sql_query = 'SELECT restaurant.name, restaurant.displayAddress, other_info.restaurant_name FROM restaurant, other_info WHERE other_info.restaurant_name = restaurant.name and categories LIKE %s and district LIKE %s and price_range LIKE %s and rating >= %s LIMIT 1' %('\'%'+slots['category']+'%\'', '\'%'+slots['area']+'%\'', '\'%' + price + '%\'', '\'' + slots['score'] + '\'')
				cursor.execute(sql_query)
				results = cursor.fetchall()
				print 'results : ' ,results
				if results.__len__() == 0:
					content = {'policy':'inform_no_match'}
					if slots['category'] != '':
						content['category'] = slots['category']
					if slots['area'] != '':
						content['area'] = slots['area']
					if slots['price'] != '':
						content['price'] = slots['price']
					if slots['score'] != '' and slots['score'] != '0.0':
						content['score'] = slots['score']
				else:
					content = {'policy':'inform_restaurant', 'name':str(results[0][0]).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"',''), 'address':str(results[0][1]).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"','')}
				#print sql_query
			
			elif intent == 'inform_address':
				cursor.execute('SELECT displayAddress FROM restaurant WHERE name LIKE %s LIMIT 1' %('\'%'+name+'%\''))
				results = cursor.fetchall()
				if results.__len__() == 0:
					content = {'policy':'inform_no_info', 'name':slots['restaurant_name'], 'info_name':'address'}
				else:
					content = {'policy':'inform_address', 'address':str(results).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"','')}
			
			elif intent == 'inform_score':
				cursor.execute('SELECT rating FROM restaurant WHERE name LIKE %s LIMIT 1' %('\'%'+name+'%\''))
				results = cursor.fetchall()
				if results.__len__() == 0:
					content = {'policy':'inform_no_info', 'name':slots['restaurant_name'], 'info_name':'score'}
				else:
					content = {'policy':'inform_score', 'score':str(results).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"','')}
			
			elif intent == 'inform_review':
				cursor.execute('SELECT review FROM reviews WHERE name LIKE %s' %('\'%'+name+'%\''))
				results = cursor.fetchall()
				if results.__len__() == 0:
					content = {'policy':'inform_no_info', 'name':slots['restaurant_name'], 'info_name':'review'}
				else:
					content = {'policy':'inform_review', 'review':str(results[random.randint(0,results.__len__()-1)]).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"','')}

			elif intent == 'inform_smoke':
				cursor.execute('SELECT Smoking FROM other_info WHERE restaurant_name LIKE %s LIMIT 1' %('\'%'+name+'%\''))
				results = cursor.fetchall()
				if results.__len__() == 0:
					content = {'policy':'inform_no_info', 'name':slots['restaurant_name'], 'info_name':'smoke'}
				else:
					out = str(results).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"','')
					if 'No' in out or out == ' ':
						content = {'policy':'inform_smoke_no'}
					else:
						content = {'policy':'inform_smoke_yes'}

			elif intent == 'inform_wifi':
				if ord(slots['restaurant_name'][0])%2 == 1:
					content = {'policy':'inform_wifi_no'}
				else:
					content = {'policy':'inform_wifi_yes'}

			elif intent == 'inform_phone':
				cursor.execute('SELECT phone FROM restaurant WHERE name LIKE %s LIMIT 1' %('\'%'+name+'%\''))
				results = cursor.fetchall()
				if results.__len__() == 0:
					content = {'policy':'inform_no_info', 'name':slots['restaurant_name'], 'info_name':'phone'}
				else:
					out = str(results).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"','')
					if 'No' in out or out == ' ':
						content = {'policy':'inform_no_info'}
					else:
						out = out.split(' ')
						content = {'policy':'inform_phone', 'phone':'('+out[0]+')'+out[1]}

			elif intent == 'inform_price':
				cursor.execute('SELECT price_range FROM other_info WHERE restaurant_name LIKE %s LIMIT 1' %('\'%'+name+'%\''))
				results = cursor.fetchall()
				if results.__len__() == 0:
					content = {'policy':'inform_no_info', 'name':slots['restaurant_name'], 'info_name':'price'}
				else:
					out = str(results).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"','')
					if out == ' ':
						content = {'policy':'inform_no_info'}
					else:
						content = {'policy':'inform_info', 'price':out}

			elif intent == 'inform_time':
				cursor.execute('SELECT open_time FROM other_info WHERE restaurant_name LIKE %s LIMIT 1' %('\'%'+name+'%\''))
				results = cursor.fetchall()
				if results.__len__() == 0:
					content = {'policy':'inform_no_info', 'name':slots['restaurant_name'], 'info_name':'time'}
				else:
					out = str(results).replace(',',' ').replace('u\'','').replace(']','').replace('[','').replace('(','').replace(')','').replace('\'','').replace('\"','')
					if out == ' ':
						content = {'policy':'inform_no_info'}
					else:
						content = {'policy':'inform_info', 'time':out}

			self.db.close()
			print content
			return content
		except MySQLdb.Error as e:
			print "Error %d: %s" % (e.args[0], e.args[1])
			return ''

if __name__ == '__main__':

	slots = {'restaurant_name':'', 'area':'NEW YORK', 'category':'british', 'score':'4.0', 'price':'$20', 'wifi':'', 'smoking':''}
  	search = SearchDB('140.112.49.151' ,'foodbot' ,'welovevivian' ,'foodbotDB')
	search.grabData('inform_restaurant' ,slots)
