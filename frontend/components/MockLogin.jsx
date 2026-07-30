'use client'

export default function MockLogin({ onLogin }) {
  return (
    <div className="flex h-screen w-full items-center justify-center bg-gray-50">
      <div className="w-full max-w-sm bg-white border border-gray-200 rounded-2xl shadow-sm p-8">
        <h1 className="text-lg font-semibold text-gray-900 mb-1 text-center">
          Kenyan Legal Assistant
        </h1>
        <p className="text-[13px] text-gray-500 text-center mb-6">
          This is a prototype — pick an account type to continue.
        </p>

        <div className="flex flex-col gap-3">
          <button
            onClick={() => onLogin('lawyer')}
            className="w-full px-4 py-3 rounded-xl border border-gray-200 hover:border-teal-300 hover:bg-teal-50 transition text-left"
          >
            <div className="text-[14px] font-medium text-gray-900">Continue as Lawyer</div>
            <div className="text-[12px] text-gray-500">Access Q&A and document drafting</div>
          </button>

          <button
            onClick={() => onLogin('normal')}
            className="w-full px-4 py-3 rounded-xl border border-gray-200 hover:border-teal-300 hover:bg-teal-50 transition text-left"
          >
            <div className="text-[14px] font-medium text-gray-900">Continue as Normal user</div>
            <div className="text-[12px] text-gray-500">Legal Q&A only</div>
          </button>
        </div>

        <p className="text-[11px] text-gray-400 text-center mt-6">
          Mock login for prototype purposes — no real account is created.
        </p>
      </div>
    </div>
  )
}
