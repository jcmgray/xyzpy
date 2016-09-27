import os

_SETTINGS_DIR_REL = ["~", ".config", "xyzpy"]
_SETTINGS_DIR = os.path.expanduser(os.path.join(*_SETTINGS_DIR_REL))
_SETTINGS_FILE = "telegram_settings.yaml"

_DEFAULT_BOTNAME = "__default_xyz_bot_for_this_dir__"


def _ensure_xyzdir_exists():
    """Ensure that the xyzpy settings folder exists.
    """
    os.makedirs(_SETTINGS_DIR, exist_ok=True)


def load_dict_from_xyzdir(fname):
    """Load a dictionary of settings from a yaml file
    in the ~/.config/xyzpy folder.
    """
    import yaml

    try:
        with open(os.path.join(_SETTINGS_DIR, fname), 'r') as f:
            settings = yaml.safe_load(f)
    except FileNotFoundError:
        settings = dict()
    return settings


def load_settings():
    return load_dict_from_xyzdir(_SETTINGS_FILE)


def save_dict_to_xyzdir(fname, settings):
    """Save a dictionary of settings to a yaml file
    in the ~/.config/xyzpy folder.
    """
    import yaml

    with open(os.path.join(_SETTINGS_DIR, fname), 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)


def save_settings(settings):
    save_dict_to_xyzdir(_SETTINGS_FILE, settings)


def delete_settings():
    try:
        os.remove(os.path.join(_SETTINGS_DIR, _SETTINGS_FILE))
    except FileNotFoundError:
        pass


def load_token(botname=None):
    """Load a bot's token, using either the default or a given botname.
    """
    if botname is None:
        botname = _DEFAULT_BOTNAME

    settings = load_settings()
    return settings["telegram_tokens"][botname]


def save_token(token, botname=None, set_as_default_bot=False):
    """Save a bot's token, using either the default or a given botname.
    """
    if botname is None:
        botname = _DEFAULT_BOTNAME

    # Load the settings file and token dict
    settings = load_settings()
    try:
        tokens_dict = settings["telegram_tokens"]
    except KeyError:
        tokens_dict = dict()
        settings["telegram_tokens"] = tokens_dict

    # Set the token for the bot and set as default if not present
    tokens_dict[botname] = token
    try:
        tokens_dict[_DEFAULT_BOTNAME]
        no_default = False
    except KeyError:
        no_default = True

    if set_as_default_bot or no_default:
        tokens_dict[_DEFAULT_BOTNAME] = token

    save_settings(settings)


def initialize_token(token=None, botname=None, save=True):
    """Initialize a bot, optionally saving to disk for automatic later use.
    """
    if token is None:
        token = load_token(botname=botname)
    elif save:  # only save new tokens
        save_token(token, botname=botname)

    return token


def initialize_bot(token):
    """
    """
    import telegram
    return telegram.Bot(token)


def get_chat_id(bot, username):
    """
    """
    msg = bot.get_updates()[-1].message
    chat_id = msg.chat_id
    from_username = msg.from_user.username
    if from_username == username:
        return chat_id


def _get_default_username():
    """
    """
    return load_settings()['__default_username__']


def load_chat_id(botname, username=None):
    """
    """
    if username is None:
        username = _get_default_username()

    settings = load_settings()
    return settings['chat_ids'][botname][username]


def save_chat_id(chat_id, botname=None, username=None,
                 set_as_default_user=False):
    """
    """
    if botname is None:
        botname = _DEFAULT_BOTNAME

    if username is None:
        username = _get_default_username()

    settings = load_settings()

    # Load the chat_ids dict of bots
    try:
        chat_ids_dict = settings["chat_ids"]
    except KeyError:
        chat_ids_dict = dict()
        settings["chat_ids"] = chat_ids_dict

    # Load the individual bots chat_ids
    try:
        bots_chat_ids_dict = chat_ids_dict[botname]
    except KeyError:
        bots_chat_ids_dict = dict()
        settings["chat_ids"][botname] = bots_chat_ids_dict

    settings["chat_ids"][botname][username] = chat_id

    # Test whether to set username as default
    try:
        settings['__default_username__']
        no_default = False
    except KeyError:
        no_default = True
    if set_as_default_user or no_default:
        settings['__default_username__'] = username


def initialize_chat_id(chat_id, botname=None, username=None, save=True):
    if botname is None:
        botname =


def send_message(bot, username, message):
    """
    """
    chat_id = load_chat_id(username)
    bot.send_message(chat_id, message)
