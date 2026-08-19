CAMERA_INDEX = 0

# Vision capture duration
CAPTURE_SECONDS = 1.5

# Number of frames per second we try to sample
TARGET_FPS = 10

# Face detection confidence
DETECTION_THRESHOLD = 0.50

# Similarity threshold for known people
KNOWN_MATCH_THRESHOLD = 0.50

# Similarity threshold for matching unknown faces
UNKNOWN_CLUSTER_THRESHOLD = 0.55

# Minimum number of observations before trusting an identity
MIN_KNOWN_OBSERVATIONS = 2
MIN_UNKNOWN_OBSERVATIONS = 2

GALLERY_PATH = "data/gallery"
UNKNOWN_PATH = "data/unknown"
MIN_FACE_SIZE = 60

BLUR_THRESHOLD = 80.0

OUTPUT_FILE = "output/results.jsonl"