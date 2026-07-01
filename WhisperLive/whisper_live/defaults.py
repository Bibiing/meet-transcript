"""Default ASR policy for the PLN WhisperLive server."""

DEFAULT_INITIAL_PROMPT = (
    "Transkrip rapat Bahasa Indonesia. Gunakan ejaan baku Bahasa Indonesia "
    "(EYD/PUEBI) hanya untuk kata yang terdengar jelas. Jangan menambah "
    "informasi, jangan mengganti makna, dan jangan menebak kreatif saat audio "
    "tidak jelas atau hening. Pertahankan istilah teknis, singkatan, nama "
    "sistem, dan kata serapan yang umum apa adanya."
)

DEFAULT_HOTWORDS = (
    "API, database, deployment, endpoint, server, staging, production, "
    "dashboard, authentication, authorization, billing, invoice, PLN, EYD, "
    "PUEBI, meteran, token listrik, pelanggan, tagihan, daya, gardu, trafo, "
    "integrasi, migrasi, aplikasi, layanan, user, admin, login, logout, "
    "repository, branch, commit, Docker, GPU, CPU"
)
