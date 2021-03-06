import math

import cv2


class TextDetect():

    def text_detect(self, _img):
        # Read and store arguments
        conf_threshold = 0.5
        nms_threshold = 0.4
        # Load network
        net = cv2.dnn.readNet('/opt/ocr/models/frozen_east_text_detection.pb')

        # get height and width of input image
        w, h = _img.shape[:2]

        # set width and height to resamnple down to (multiples of 32)
        width = 640
        height = 640

        r_w: float = w / float(width)
        r_h: float = h / float(height)

        # set names of features to forward from net
        out_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # Create a 4D blob from frame.
        blob = cv2.dnn.blobFromImage(_img, 1.0, (width, height), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        outs = net.forward(out_names)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = self.decode(scores, geometry, conf_threshold)

        print(boxes, confidences)

        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= r_w
                vertices[j][1] *= r_h
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                print(p1, p2)
                cv2.line(_img, p1, p2, (0, 0, 0), 1)

        # Put efficiency information
        cv2.putText(_img, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        return _img

    def decode(self, scores, geometry, scoreThresh):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scores_data = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            angles_data = geometry[0][4][y]
            for x in range(0, width):
                score = scores_data[x]

                # If score is lower than threshold score, move to next x
                if (score < scoreThresh):
                    continue

                # Calculate offset
                offset_x = x * 4.0
                offset_y = y * 4.0
                angle = angles_data[x]

                # Calculate cos and sin of angle
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = (
                    [offset_x + cos_a * x1_data[x] + sin_a * x2_data[x], offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]])

                # Find points for rectangle
                p1 = (-sin_a * h + offset[0], -cos_a * h + offset[1])
                p3 = (-cos_a * w + offset[0], sin_a * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]
