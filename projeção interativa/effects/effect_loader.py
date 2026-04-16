import os
import sys
import importlib.util
import inspect

class EffectLoader:
    """Carrega e gerencia efeitos, jogos e visualizações imersivas"""
    
    def __init__(self, effects_dir="effects"):
        self.effects_dir = effects_dir
        self.effects = {}
        self.categories = {
            "simple": {},
            "games": {},
            "immersive": {},
            "future": {}
        }
    
    def load_all_effects(self):
        """Carrega todos os efeitos das pastas simples, games, immersive e future"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        for category in ["simple", "games", "immersive", "future"]:
            category_path = os.path.join(base_path, category)
            if not os.path.exists(category_path):
                print(f"[EffectLoader] Pasta {category} não encontrada, criando...")
                os.makedirs(category_path, exist_ok=True)
                continue
            
            # Carrega todos os arquivos .py da categoria (exceto __init__.py)
            for filename in os.listdir(category_path):
                if filename.endswith('.py') and filename != '__init__.py':
                    effect_name = filename[:-3]  # Remove .py
                    self._load_effect(category, category_path, effect_name)
        
        print(f"[EffectLoader] Carregados {len(self.effects)} efeitos:")
        for category, effects in self.categories.items():
            print(f"  {category}: {len(effects)} efeitos")
    
    def _load_effect(self, category, category_path, effect_name):
        """Carrega um efeito individual"""
        try:
            file_path = os.path.join(category_path, f"{effect_name}.py")
            spec = importlib.util.spec_from_file_location(effect_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Encontra a classe do efeito (deve terminar com 'Effect' ou ser a única classe)
            effect_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name.endswith('Effect'):
                    effect_class = obj
                    break
            
            if effect_class is None:
                print(f"[EffectLoader] Aviso: Nenhuma classe Effect encontrada em {effect_name}.py")
                return
            
            # Instancia o efeito
            effect_instance = effect_class()
            
            # Verifica se tem os atributos necessários
            if not hasattr(effect_instance, 'effect_id'):
                print(f"[EffectLoader] Aviso: {effect_name} não tem effect_id")
                return
            
            # Armazena o efeito
            self.effects[effect_instance.effect_id] = effect_instance
            self.categories[category][effect_instance.effect_id] = effect_instance
            
            print(f"[EffectLoader] Carregado: {effect_instance.name} ({category})")
            
        except Exception as e:
            print(f"[EffectLoader] Erro ao carregar {effect_name}: {e}")
    
    def get_effect(self, effect_id):
        """Retorna uma instância do efeito pelo ID"""
        return self.effects.get(effect_id)
    
    def get_effects_by_category(self, category):
        """Retorna todos os efeitos de uma categoria"""
        return self.categories.get(category, {})
    
    def get_all_effects(self):
        """Retorna todos os efeitos"""
        return self.effects
    
    def get_effect_display_name(self, effect_id):
        """Retorna o nome de exibição do efeito"""
        effect = self.get_effect(effect_id)
        if effect:
            return effect.name
        return "Desconhecido"
    
    def reload_effects(self):
        """Recarrega todos os efeitos"""
        self.effects = {}
        self.categories = {
            "simple": {},
            "games": {},
            "immersive": {},
            "future": {}
        }
        self.load_all_effects()
