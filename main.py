from langchain_community.document_loaders import YoutubeLoader
video_url = "https://www.youtube.com/watch?v=J5_-l7WIO_w&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=17"

loader = YoutubeLoader.from_youtube_url(
    video_url,
    add_video_info=False,
    language=["hi"]
)

docs = loader.load()
transcript = " ".join(doc.page_content for doc in docs)
print("Transcript loaded successfully!")
print(transcript)
for i, doc in enumerate(docs, start=1):
    print(f"\n--- Chunk {i} ---")
    print("Metadata:", doc.metadata)
    print("Transcript:")
    print(doc.page_content)