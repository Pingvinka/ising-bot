import numpy as np
import torch
import torch.nn as nn
import math
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
import io
import hashlib
import nest_asyncio
import os
import sys

# для работы в Jupyter/Colab
nest_asyncio.apply()

# Настройка логирования
import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def set_seeds(seed=42): #Сиды
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

class AdvancedGNN(nn.Module):
    def __init__(self, n_nodes, hidden=256):
        super().__init__()
        self.n_nodes = n_nodes

        self.encoder = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, 1)
        )

        self.stop_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.n_nodes, -1)

        h = self.encoder(x)
        global_emb = h.mean(dim=1)

        global_expanded = global_emb.unsqueeze(1).expand(-1, self.n_nodes, -1)
        node_features = torch.cat([h, global_expanded], dim=-1)
        node_logits = self.policy_head(node_features).squeeze(-1)

        stop_logit = self.stop_head(global_emb).squeeze(-1)
        logits = torch.cat([node_logits, stop_logit.unsqueeze(1)], dim=1)
        value = self.value_head(global_emb).squeeze(-1)

        return logits, value

class AdvancedIsingEnv:
    def __init__(self, adj_matrix, seed=42):
        self.J = adj_matrix.astype(np.float32)
        self.N = len(adj_matrix)
        self.np_random = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.spins = self.np_random.choice([-1, 1], self.N)
        self.best_spins = self.spins.copy()
        self.best_energy = self.energy(self.spins)
        self.steps = 0
        self.no_improvement_steps = 0
        self.consecutive_flips = np.zeros(self.N, dtype=int)
        return self._get_obs()

    def step(self, action):
        self.steps += 1
        self.consecutive_flips += 1

        if action == self.N: 
            return self._get_obs(), 0, True, {
                "energy": self.energy(self.spins),
                "best_energy": self.best_energy
            }

        self.consecutive_flips[action] = 0
        old_energy = self.energy(self.spins)
        self.spins[action] *= -1
        new_energy = self.energy(self.spins)

        reward = old_energy - new_energy

        if new_energy < self.best_energy:
            self.best_spins = self.spins.copy()
            self.best_energy = new_energy
            self.no_improvement_steps = 0
            reward += 1.0
        else:
            self.no_improvement_steps += 1
            reward -= 0.1

        done = (self.steps >= self.N * 3 or
                self.no_improvement_steps >= self.N * 2)

        return self._get_obs(), reward, done, {
            "energy": new_energy,
            "best_energy": self.best_energy
        }

    def energy(self, spins):
        return -0.5 * np.sum(self.J * np.outer(spins, spins))

    def _get_obs(self):
        local_field = self.J @ self.spins
        delta_energy = 2 * self.spins * local_field
        deg_pos = (self.J == 1).sum(axis=1)
        deg_neg = (self.J == -1).sum(axis=1)

        return np.column_stack([
            self.spins,
            local_field,
            delta_energy,
            deg_pos / self.N,
            deg_neg / self.N,
            self.consecutive_flips / 10.0
        ]).astype(np.float32)

def predict_spins_deterministic(agent, adj_matrix, n_restarts=10):
    best_energy = float('inf')
    best_spins = None

    # Фиксированные сиды для каждого рестарта
    seeds = [42 + i * 100 for i in range(n_restarts)]

    for restart, seed in enumerate(seeds):
        set_seeds(seed)

        env = AdvancedIsingEnv(adj_matrix, seed=seed)
        state = env.reset()
        local_best_spins = env.best_spins.copy()
        local_best_energy = env.best_energy

        for step in range(env.N * 30):
            state_t = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                logits, _ = agent(state_t)
                action = torch.argmax(logits, dim=-1).item()

            state, _, done, info = env.step(action)

            if info['best_energy'] < local_best_energy:
                local_best_energy = info['best_energy']
                local_best_spins = env.best_spins.copy()

            if done or action == env.N:
                break

        if local_best_energy < best_energy:
            best_energy = local_best_energy
            best_spins = local_best_spins
    print(best_energy)
    return best_spins

def load_model(model_path, n_spins):
    try:
        logger.info(f"🔄 Attempting to load model from: {model_path}")
        
        # Проверяем существование файла
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            logger.info(f"📁 Current directory: {os.getcwd()}")
            logger.info(f"📁 Directory contents: {os.listdir('.')}")
            if os.path.exists('models'):
                logger.info(f"📁 Models directory contents: {os.listdir('models')}")
            return None
            
        logger.info("✅ Model file exists")
        
        # Пробуем загрузить модель
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            logger.info("✅ Model loaded with weights_only=False")
        except Exception as e1:
            logger.warning(f"⚠️ First load attempt failed: {e1}")
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                logger.info("✅ Model loaded with default parameters")
            except Exception as e2:
                logger.error(f"❌ Error loading model: {e2}")
                return None
        
        # Создаем агента
        agent = AdvancedGNN(n_spins)
        logger.info("✅ AdvancedGNN model created")
        
        # Проверяем структуру checkpoint
        logger.info(f"📁 Checkpoint keys: {list(checkpoint.keys())}")
        
        # Загружаем веса
        if 'agent' in checkpoint:
            agent.load_state_dict(checkpoint['agent'])
            logger.info("✅ Model weights loaded from 'agent' key")
        elif 'model_state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Model weights loaded from 'model_state_dict' key")
        elif 'state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['state_dict'])
            logger.info("✅ Model weights loaded from 'state_dict' key")
        else:
            try:
                agent.load_state_dict(checkpoint)
                logger.info("✅ Model weights loaded directly from checkpoint")
            except Exception as e:
                logger.error(f"❌ Could not load model weights: {e}")
                return None
        
        agent.eval()
        logger.info("✅ Model set to eval mode")
        return agent
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

def read_matrix_from_file(file_content):
    lines = file_content.decode('utf-8').strip().split('\n')
    matrix = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]
        matrix.append(row)
    return np.array(matrix)

def save_spins_to_file(spins):
    output = io.StringIO()
    for spin in spins:
        output.write(f"{spin}\n")
    return output.getvalue()

results_cache = {}

def get_matrix_hash(matrix):
    return hashlib.md5(matrix.tobytes()).hexdigest()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для решения задачи Изинга :>\n"
        "Отправьте мне файл .txt с матрицей смежности, и я верну оптимальные спины.\n\n"
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Проверяем, что модель загружена
        if 'agent' not in context.bot_data or context.bot_data['agent'] is None:
            await update.message.reply_text("❌ Модель не загружена. Бот не может обрабатывать файлы.")
            return
            
        file = await update.message.document.get_file()
        file_content = await file.download_as_bytearray()

        await update.message.reply_text("Обрабатываю матрицу...")

        adj_matrix = read_matrix_from_file(file_content)
        n_spins = len(adj_matrix)

        expected_n_spins = context.bot_data['n_spins']
        if n_spins != expected_n_spins:
            await update.message.reply_text(
                f"Размер матрицы ({n_spins}) не соответствует ожидаемому "
                f"({expected_n_spins}). Пожалуйста, проверьте данные."
            )
            return

        matrix_hash = get_matrix_hash(adj_matrix)
        if matrix_hash in results_cache:
            spins = results_cache[matrix_hash]
            cache_info = " (из кэша)"
        else:
            agent = context.bot_data['agent']
            spins = predict_spins_deterministic(agent, adj_matrix, n_restarts=8)
            results_cache[matrix_hash] = spins
            cache_info = ""

        result_text = save_spins_to_file(spins)

        result_file = io.BytesIO(result_text.encode('utf-8'))
        result_file.name = f"spins_{n_spins}.txt"

        energy = -0.5 * np.sum(adj_matrix * np.outer(spins, spins))

        await update.message.reply_text(
            f"Решение готово{cache_info}!\n"
            f"Энергия: {energy:.2f}\n"
        )
        await update.message.reply_document(document=result_file)

    except Exception as e:
        logger.error(f"Error handling file: {e}")
        await update.message.reply_text(f"❌ Ошибка при обработке файла: {str(e)}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Инструкция по использованию:\n\n"
        "1. Подготовьте файл .txt с симметричной матрицей смежности\n"
        "2. Матрица должна быть размером 200×200\n"
        "3. Каждая строка должна содержать числа, разделенные пробелами\n"
        "4. Отправьте файл боту\n\n"
        "Пример формата файла:\n"
        "0 1 -1 0 ...\n"
        "1 0 0 -1 ...\n"
        "-1 0 0 1 ...\n"
        "0 -1 1 0 ...\n"
        "...\n\n"
      
        "Команды:\n"
        "/start - начать работу\n"
        "/help - показать эту справку\n"
        "/tea - выпить чай ☕"
    )

async def clear_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global results_cache
    old_size = len(results_cache)
    results_cache = {}
    await update.message.reply_text(f"🧹 Кэш очищен! Удалено {old_size} записей.")

async def tea_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_path = "assets/peite_chai.jpg"  
        
        if not os.path.exists(photo_path):
            await update.message.reply_text("К сожалению, чайник сломался... Фотография не найдена!")
            return
        
        with open(photo_path, 'rb') as photo:
            await update.message.reply_photo(
                photo=photo,
                caption="☕ Вот ваш чай! Приятного чаепития! 🍵\n\n"
                       "Пока пьёте чай, можете отправить мне матрицу для решения задачи Изинга! 😊"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Чайник закипел с ошибкой: {str(e)}")

async def main_async():
    try:
        logger.info("🚀 Starting bot initialization...")
        
        TOKEN = "8481020311:AAFtFAzgahTdfX3kB3uA1ySefGFtn6_VjYk"  
        MODEL_PATH = "models/best_ising_model_ppg.pth"
        N_SPINS = 200

        logger.info(f"🔧 Configuration: TOKEN={TOKEN[:10]}..., MODEL_PATH={MODEL_PATH}, N_SPINS={N_SPINS}")

        # Проверяем файловую систему
        logger.info(f"📁 Current working directory: {os.getcwd()}")
        logger.info(f"📁 Directory contents: {os.listdir('.')}")
        
        if os.path.exists('models'):
            logger.info(f"📁 Models directory contents: {os.listdir('models')}")
        else:
            logger.warning("⚠️ Models directory does not exist!")

        logger.info("🔄 Loading model...")
        agent = load_model(MODEL_PATH, N_SPINS)
        
        if agent is None:
            logger.error("❌ Failed to load model. Bot cannot start.")
            return

        logger.info("✅ Model loaded successfully!")
        
        logger.info("🔧 Creating bot application...")
        application = Application.builder().token(TOKEN).build()

        application.bot_data['agent'] = agent
        application.bot_data['n_spins'] = N_SPINS

        # Добавляем обработчики
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("clear_cache", clear_cache))
        application.add_handler(CommandHandler("tea", tea_command))
        application.add_handler(MessageHandler(filters.Document.TXT, handle_file))

        logger.info("🤖 Starting bot polling...")
        await application.run_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES,
            timeout=30
        )
        
    except Exception as e:
        logger.error(f"💥 Critical error in main_async: {e}")
        raise

if __name__ == "__main__":
    logger.info("🎯 Script started")
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
    finally:
        logger.info("🏁 Script finished")
