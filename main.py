import io
import os
import random

import numpy as np
import onnxruntime as ort
import telebot
from dotenv import load_dotenv
from PIL import Image
from telebot import types

from utils import CLASSES, HINTS, MESSAGE_TEXTS, PATH_TO_MODEL, TRANSFORM

load_dotenv()

tg_token = os.getenv('tg_token')
bot = telebot.TeleBot(tg_token, num_threads=4)

# menu buttons
bot.set_my_commands([
    types.BotCommand('/hint', 'Хочу подсказку по сортировке!'),
    types.BotCommand('/help', 'Инструкция'),
])


@bot.message_handler(commands=['start', 'help'])
def start(message):
    """
    Sends a start message
    """
    bot.send_message(
        message.chat.id,
        text=start_text,
        parse_mode='html'
    )


@bot.message_handler(content_types=['photo'])
def predict_class(message):
    """
    Receives a photo, sends back a model prediction.
    Sends a message, then edits the message with the prediction.
    """
    message_to_edit = bot.send_message(
        message.chat.id,
        text=random.choice(MESSAGE_TEXTS)
    )

    try:
        # downloading
        file_info = bot.get_file(message.photo[-1].file_id)
        dwn_file = bot.download_file(file_info.file_path)
        # image preprocessing
        byte_obj = io.BytesIO(dwn_file)
        img = Image.open(byte_obj)
        img = np.array(img, dtype=np.float32)
        img = TRANSFORM(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        # model inference
        ort_inputs = {'input': img[None, ...]}
        ort_outs = ort_session.run(None, ort_inputs)
    
    except Exception as e:
        print(e)

        bot.edit_message_text(
            text='К сожалению, что-то пошло не так. Попробуй еще раз!',
            chat_id=message.chat.id,
            message_id=message_to_edit.message_id,
        )
    
    else:
        bot.edit_message_text(
            text=f'Предсказанный класс: <b>{CLASSES[ort_outs[0].argmax()]}</b>',
            chat_id=message.chat.id,
            message_id=message_to_edit.message_id,
            parse_mode='html'
        )


@bot.message_handler(commands=['hint'])
def hint(message):
    """
    Randomly chooses and sends a text
    from the texts/hints directory
    """
    file_path = random.choice(HINTS)
    with open(file_path, 'r') as f:
        to_send = f.read()

    bot.send_message(
        message.chat.id,
        text=to_send,
        parse_mode='html'
    )


@bot.message_handler(content_types=['text'])
def get_user_text(message):
    """
    Arbitrary text message handling + an easter egg
    """
    if message.text in ['❤️', '❤️❤️', '❤️❤️❤️']:
        to_send = '❤️' * random.choice([1, 3])

    else:
        to_send = 'К сожалению, я тебя не понимаю '
        to_send += random.choice(['😿', '😭', '😤', '😇', '😅'])
        to_send += '\n\nОзнакомиться с инструкцией можно в меню!'

    bot.send_message(
        message.chat.id,
        text=to_send
    )


if __name__ == '__main__':
    with open('texts/start.txt', 'r') as f:
        start_text = f.read()

    ort_session = ort.InferenceSession(
        PATH_TO_MODEL,
        providers=["CPUExecutionProvider"]
    )

    bot.infinity_polling()
