def main():

    parser = argparse.ArgumentParser(
        description="Boson – Vocalis audio generation using HiggsAudio model"
    )

    parser.add_argument(
        "voice",
        help="Voice sample name to use (e.g. 'voice', 'chris', 'narrator_dark')",
        nargs = "?",  # makes it optional
        default = None,  # default value if not provided
    )

    args = parser.parse_args()

    voice = args.voice
    samples = [
        # Anchor / tonal conditioning sample (generate first, do NOT trim)
        "Take a steady breath. The cold waits for no one.",
        # Follow-ups (generate separately, appended after anchor in playback)
        # "The air is thin, and it settles softly in your lungs.",
        # "Each step carries weight, and the mountain remembers them all.",
        # "Frost gathers along the edges of silence.",
        # "Somewhere in the distance, something is watching.",
        # "The wind moves through stone and bone alike.",
        # "The world is vast, indifferent, and older than your fear."
    ]

    sync_voice_prompts()

    logger.info(f"Using voice sample: {voice}")

    text = ""
    for sample in samples:
        text += sample + "\n"
        logger.info(f"Sample ({len(sample)} chars): {sample}...")

    chunked_text = chunker.chunk_text(text)

    device = get_device("auto")

    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device=device)

    model_client = get_model_client()

    speaker_tags = []

    messages, audio_ids = prepare_generation_context(
        scene_prompt="",
        ref_audio=voice,
        ref_audio_in_system_message=True,
        audio_tokenizer=audio_tokenizer,
        speaker_tags=speaker_tags,
    )

    # if voice is None:
    #
    #     messages[0] = Message(
    #         role="system",
    #         content=(
    #             # "You are a deep, sonorous narrator. "
    #             # "Your voice is cold, restrained, and heavy with foreboding. "
    #             # "Each sentence carries weight. "
    #             # "Pause slightly between phrases. "
    #             # "Emotion is subtle but unmistakable."
    #             "You are a low, warm narrator."
    #             "Speech is steady, controlled, and deliberate."
    #             "Emotion is restrained."
    #             "Consonants are soft."
    #             "Pacing is moderate."
    #         ),
    #     )

    concat_wv, sr, text_output = model_client.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        generation_chunk_buffer_size=2,
        temperature=0.65,
        top_k=80,
        top_p=0.97,
        ras_win_len=4 + 20,
        ras_win_max_num_repeat=3 + 1,
        seed=1001,
    )

    sf.write("boson.wav", concat_wv, sr)

if __name__ == "__main__":
    main()
