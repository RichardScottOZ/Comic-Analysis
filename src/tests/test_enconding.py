import urllib.parse

def encode_s3_uri(s3_uri):
    """Encodes spaces and special characters in an S3 URI while preserving s3:// and slashes."""
    if not s3_uri.startswith('s3://'):
        return s3_uri
    
    # Split s3://bucket/key
    parts = s3_uri[5:].split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    # Encode only the key part, but preserve slashes
    encoded_key = urllib.parse.quote(key, safe='/')
    
    return f"s3://{bucket}/{encoded_key}"

test_uri = "s3://bucket/NeonIchiban/THE_DEVIL'S_CUT/page001.jpg"
encoded = encode_s3_uri(test_uri)
print(f"Original: {test_uri}")
print(f"Encoded : {encoded}")

# Manual fix test
fixed_key = urllib.parse.quote(test_uri[5:].split('/', 1)[1], safe='/').replace("'", "%27")
print(f"Fixed   : s3://bucket/{fixed_key}")
