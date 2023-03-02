class BoundingBox:
    """
    Class used to store the bounding box results of the YOLO output
    These can be cleanly handled by other files rather than a set of tuples
    """
    
    def __init__(self, x, y, w, h, score, class_name, timestamp):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.class_name = class_name
        self.timestamp = timestamp

    def get_corners(self):
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

    def get_mid_point(self):
        """
        Gets the midpoint of the bounding box, returns coordinates as (x, y)
        """
        midx = self.x + round(self.w / 2)
        midy = self.y + round(self.h / 2)

        return [midx, midy]

    def get_class_and_score(self):
        """
        Gets the classID and confidence score attributes, returns as an array [classID, confidence score]
        """

        return [self.class_name, self.score]

    def get_JSON_dict(self) -> dict:
        """Gets a dictionary of useful features which can be turned into
        JSON string

        Returns:
            dict: JSON dictionary with useful details
        """

        json_dict = {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.w),
            "height": int(self.h),
            "score": float(self.score),
            "class": self.class_name
        }

        return json_dict

        return json_dict

    def print_BB(self):
        """
        Provides a formatted method for printing the bounding box class
        This method is useful for debugging
        """
        print(self.x, self.y, self.w, self.h, self.score, self.class_name, self.timestamp)
