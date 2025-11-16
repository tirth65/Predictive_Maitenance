import React from 'react';

const StatusCard = ({ title, value, icon: Icon, color = "blue" }) => (
  <div className="bg-white rounded-lg p-6 border border-gray-200">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-gray-600 text-sm">{title}</p>
        <p className="text-2xl font-semibold text-gray-900">{value}</p>
      </div>
      <Icon className={`h-8 w-8 text-${color}-500`} />
    </div>
  </div>
);

export default StateCard;