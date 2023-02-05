class Bounding_Box:
    """
    Class used to store the bounding box results of the YOLO output
    These can be cleanly handled by other files rather than a set of tuples
    """
    
    def __init__(self, x, y, w, h, score, className, timestamp):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.className = className
        self.timestamp = timestamp

    def getCorners(self):
        """
        Gets the four corners of the bounding box starting in the top left in a clockwise direction.
        Coordinates are in (x, y) format
        """
        
        corners = []
        corners.append([self.x, self.y])
        corners.append([self.x + self.w, self.y])
        corners.append([self.x + self.w, self.y + self.h])
        corners.append([self.x, self.y + self.h])

        return corners

    def getMidPoint(self):
        """
        Gets the midpoint of the bounding box, returns coordinates as (x, y)
        """
        midx = self.x + round(self.w / 2)
        midy = self.y + round(self.h / 2)

        return [midx, midy]

    def getClassAndScore(self):
        """
        Gets the classID and confidence score attributes, returns as an array [classID, confidence score]
        """

        return [self.className, self.score]

    def printBB(self):
        """
        Provides a formatted method for printing the bounding box class
        This method is useful for debugging
        """
        print(self.x, self.y, self.w, self.h, self.score, self.className, self.timestamp)