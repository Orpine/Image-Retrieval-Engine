import image_retrival_engine, cv2

if __name__ == "__main__":
	engine = image_retrival_engine.engine()
	#engine.build()
	engine.encode()
	#engine.init(True)
	#res = engine.predict(cv2.imread('test.jpg'), True, 'euclidean')
	#print res
