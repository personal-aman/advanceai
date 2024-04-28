def create_overlapping_segments(text, seg_length, overlap):
    # Split the text into words
    words = text.split()

    # Initialize segments and start position
    segments = []
    start = 0

    # Loop through the text and create segments
    while start < len(words):
        # End position is start plus segment length
        end = start + seg_length

        # Append the segment from start to end
        segments.append(' '.join(words[start:end]))

        # Update start position with segment length minus overlap
        start += (seg_length - overlap)

    return segments